#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <torch/library.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

template <typename T>
__host__ static inline T *get_data_ptr(py::object tensor)
{
    return reinterpret_cast<T *>(tensor.attr("data_ptr")().cast<uintptr_t>());
}

constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 4;
constexpr int SWIZZLE_WIDTH = 64;
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

__host__ __device__ __forceinline__
constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }

#define CUDA_CHECK(err) \
    do { \
        cudaError_t err_ = (err); \
        if (err_ != cudaSuccess) { \
            printf("CUDA error %d at %s:%d: %s\n", err_, __FILE__, __LINE__, \
                   cudaGetErrorString(err_)); \
        } \
    } while (0)

inline void check_cu(CUresult err)
{
  if (err == CUDA_SUCCESS) return;
  const char *msg;
  if (cuGetErrorString(err, &msg) != CUDA_SUCCESS) msg = "unable to get error string";
  TORCH_CHECK(false, msg);
}

inline void check_cuda(cudaError_t err)
{
  if (err == cudaSuccess) return;
  TORCH_CHECK(false, cudaGetErrorString(err));
}

__device__ __forceinline__
uint32_t elect_sync()
{
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}"
        : "+r"(pred)
        : "r"(0xFFFFFFFF)
    );
    return pred;
}

__device__ __forceinline__
void mbarrier_init(int mbarrier_address, int count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbarrier_address), "r"(count));
}

// https://github.com/NVIDIA/cutlass/blob/v4.2.1/include/cutlass/arch/barrier.h#L408
__device__ __forceinline__
void mbarrier_wait(int mbar_addr, int phase) {
  uint32_t ticks = 0x989680;  // this is optional
  asm volatile(
    "{\n\t"
    ".reg .pred P1;\n\t"
    "LAB_WAIT:\n\t"
    "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
    "@P1 bra.uni DONE;\n\t"
    "bra.uni LAB_WAIT;\n\t"
    "DONE:\n\t"
    "}"
    :: "r"(mbar_addr), "r"(phase), "r"(ticks)
  );
}

__device__ __forceinline__
void tma_2d_gmem2smem(int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
              :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(mbar_addr) : "memory");
}

template <int CTA_GROUP = 1>
__device__ __forceinline__
void tma_3d_gmem2smem(int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
  // when CTA_GROUP=1, we can use .shared::cta instead.
  // but .shared::cluster doesn't seem to be slower, so always use it unconditionally here.
  // .cta_group::2 allows mbar_addr and dst to be in different CTA's smem.
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::%6 "
              "[%0], [%1, {%2, %3, %4}], [%5];"
              :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(mbar_addr), "n"(CTA_GROUP)
              : "memory");
}

template<int CTA_GROUP = 2>
__device__ __forceinline__
void tma_3d_multicast_gmem2smem(int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr, uint16_t ctaMask) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::%7 "
        "[%0], [%1, {%2, %3, %4}], [%5], %6;"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(mbar_addr), "h"(ctaMask), "n"(CTA_GROUP)
        : "memory");
}

// Raw gmem→smem bulk copy at an arbitrary address (no TMA descriptor).
// Data is copied byte-for-byte, so if the source gmem region is already in
// a swizzled byte layout (e.g. produced by a prior TMA-descriptor load then
// smem→gmem dump), the destination smem ends up with the same swizzled
// layout that tcgen05.mma descriptors expect. nbytes must be a multiple of
// 16 and the gmem/smem addresses must be 16-byte aligned.
__device__ __forceinline__
void cp_async_bulk_g2s_raw(int smem_dst, const void* gmem_src, int nbytes, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
        "[%0], [%1], %2, [%3];"
        :: "r"(smem_dst), "l"(gmem_src), "r"(nbytes), "r"(mbar_addr)
        : "memory");
}

// Same raw g2s bulk copy, but completion is tracked via bulk_group (not
// mbarrier). Use cp.async.bulk.commit_group / wait_group<N> to synchronize
// instead of an mbar. This avoids the cross-CTA mbar-signaling issue that
// the mbarrier variant has with cluster-mapped mbar addresses on B200.
__device__ __forceinline__
void cp_async_bulk_g2s_bulk_group(int smem_dst, const void* gmem_src, int nbytes) {
    asm volatile(
        "cp.async.bulk.shared::cta.global.bulk_group "
        "[%0], [%1], %2;"
        :: "r"(smem_dst), "l"(gmem_src), "r"(nbytes)
        : "memory");
}

// commit_group / wait_group for bulk copies (works for both loads and stores).
// tma_store_wait<N>() already wraps wait_group for stores; these helpers are
// equivalent but named for use in producer load paths.
__device__ __forceinline__
void cp_async_bulk_commit_group_load() {
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

template<int N>
__device__ __forceinline__
void cp_async_bulk_wait_group_load() {
    asm volatile("cp.async.bulk.wait_group %0;" :: "n"(N) : "memory");
}

__device__ __forceinline__
constexpr uint64_t desc_encode(uint64_t x) { return (x & 0x3'FFFFULL) >> 4ULL; };

inline void init_tmap_2d(
  CUtensorMap *tmap,
  const nv_bfloat16 *ptr,
  uint64_t global_height, uint64_t global_width,
  uint32_t shared_height, uint32_t shared_width,
  CUtensorMapSwizzle swizzle
)
{
  constexpr uint32_t rank = 2;
  uint64_t globalDim[rank]       = {global_width, global_height};
  uint64_t globalStrides[rank-1] = {global_width * sizeof(nv_bfloat16)};  // in bytes
  uint32_t boxDim[rank]          = {shared_width, shared_height};
  uint32_t elementStrides[rank]  = {1, 1};

  auto err = cuTensorMapEncodeTiled(
    tmap,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    rank,
    (void *)ptr,
    globalDim,
    globalStrides,
    boxDim,
    elementStrides,
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    swizzle,
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  );
  check_cu(err);
}

inline void init_3d_tma_map(CUtensorMap *tmap, const __nv_bfloat16 *ptr, int K, int BK, uint64_t global_height, uint32_t shared_height, CUtensorMapSwizzle swizzle,
                            CUtensorMapL2promotion l2_promo = CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE)
{
	constexpr uint32_t rank = 3;
    uint64_t globalDim[rank]       = {64, global_height, (uint64_t)K / 64};
    uint64_t globalStrides[rank-1] = {(uint64_t)K * sizeof(nv_bfloat16), 128};  // in bytes
    uint32_t boxDim[rank]          = {64, shared_height, (uint32_t)BK / 64};
    uint32_t elementStrides[rank]  = {1, 1, 1};

    auto err = cuTensorMapEncodeTiled(
    	tmap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        rank,
        (void *)ptr,
        globalDim,
        globalStrides,
        boxDim,
        elementStrides,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle, //CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B
        l2_promo,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    check_cu(err);
}

template<int CTA_GROUP = 1>
__device__ __forceinline__
void tcgen05_mma_bf16(int tmem_address, uint64_t a_descr, uint64_t b_descr, uint32_t i_descr, int enable_input_d)
{
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::%5.kind::f16 [%0], %1, %2, %3, p;\n\t" // %5 is index of cta_group
        "}" 
        :: "r"(tmem_address), "l"(a_descr), "l"(b_descr), "r"(i_descr), "r"(enable_input_d), "n"(CTA_GROUP)
    );
}

__device__ __forceinline__
uint64_t make_smem_desc(int addr)
{
	const int stride_byte_offset = 8 * SWIZZLE_WIDTH * sizeof(__nv_bfloat16); // row * col, 128B swizzle pattern repeats every 8 rows
	return desc_encode(addr) | (desc_encode(stride_byte_offset) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
}

__device__ __forceinline__
void mbarrier_arrive_expect(const int mbarrier_address, const int copy_size)
{
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                :: "r"(mbarrier_address), "r"(copy_size) : "memory");
}

// cluster-scoped arrive_expect — for signaling a barrier in any CTA's smem
__device__ __forceinline__
void mbarrier_arrive_expect_cluster(const int mbarrier_address, const int copy_size)
{
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                :: "r"(mbarrier_address), "r"(copy_size) : "memory");
}

// cluster-scoped arrive (no tx) — for signaling a barrier in any CTA's smem
__device__ __forceinline__
void mbarrier_arrive_cluster(const int mbarrier_address)
{
    asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                :: "r"(mbarrier_address) : "memory");
}

// remap a shared memory address to another CTA's address space within the cluster
__device__ __forceinline__
uint32_t map_smem_addr_to_cta_rank(uint32_t smem_addr, uint32_t target_rank)
{
    uint32_t result;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(result) : "r"(smem_addr), "r"(target_rank));
    return result;
}

__device__ __forceinline__
void cluster_fence_mbarrier_init()
{
    asm volatile("fence.mbarrier_init.release.cluster;");
}

// https://github.com/NVIDIA/cutlass/blob/acb45938e9cb3e4db8c1d75155b63d31791e0e5d/include/cute/arch/cluster_sm90.hpp#L158
__device__ __forceinline__
uint32_t get_cluster_cta_rank()
{
    uint32_t rank;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(rank));
    return rank;
}

__device__ __forceinline__
void cluster_sync()
{
    asm volatile("barrier.cluster.arrive.release.aligned;\n"
                 "barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}

template<int CTA_GROUP = 1>
__device__ __forceinline__
void alloc_tmem(const int tmem_addr, int width)
{
    asm volatile("tcgen05.alloc.cta_group::%2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(tmem_addr), "r"(width), "n"(CTA_GROUP));
}

template<int CTA_GROUP = 1>
__device__ __forceinline__
void tcgen05_commit(const int mbarrier_address)
{
    asm volatile("tcgen05.commit.cta_group::%1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                        :: "r"(mbarrier_address), "n"(CTA_GROUP) : "memory");
}

template<int CTA_GROUP = 2>
__device__ __forceinline__
void tcgen05_commit_multicast(const int mbarrier_address, uint16_t ctaMask)
{
    asm volatile("tcgen05.commit.cta_group::%2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                        :: "r"(mbarrier_address), "h"(ctaMask), "n"(CTA_GROUP) : "memory");
}

__device__ __forceinline__
void tcgen05_sync()
{
    asm volatile("tcgen05.fence::after_thread_sync;");
}

__device__ __forceinline__
void tcgen05_ld(float *tmp, int addr)
{
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]),
                      "=f"(tmp[4]), "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7])
                    : "r"(addr));
}

__device__ __forceinline__
void tcgen05_wait_ld()
{
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}

__device__ __forceinline__
void tcgen05_before_thread_sync()
{
    asm volatile("tcgen05.fence::before_thread_sync;");
}

// TMA store: shared memory → global memory
__device__ __forceinline__
void tma_store_fence()
{
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__
void tma_2d_smem2gmem(int src, const void *tmap_ptr, int x, int y)
{
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
                :: "l"(tmap_ptr), "r"(src), "r"(x), "r"(y) : "memory");
}

__device__ __forceinline__
void tma_store_commit()
{
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

template<int N>
__device__ __forceinline__
void tma_store_wait()
{
    asm volatile("cp.async.bulk.wait_group %0;" :: "n"(N) : "memory");
}

// Named barrier for partial threadblock synchronization
__device__ __forceinline__
void barrier_sync(int bar_id, int thread_count)
{
    asm volatile("bar.sync %0, %1;" :: "r"(bar_id), "r"(thread_count));
}

// stmatrix.sync.aligned.x4.m8n8.shared.b16 — stores 4 × 8x8 bf16 matrices from
// 32 threads × 4×b32 regs. Per-thread addr points to row (lane%8) of matrix
// (lane/8). HW-permuted internally to avoid bank conflicts on the matched
// 128B-swizzled smem layout (same swizzle the TMA store descriptor expects).
__device__ __forceinline__
void stmatrix_x4(int smem_addr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3)
{
    asm volatile(
        "stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};"
        :: "r"(smem_addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3)
    );
}

template<int CTA_GROUP = 1>
__device__ __forceinline__
void dealloc_tmem(int taddr, int width)
{
    asm volatile("tcgen05.dealloc.cta_group::%2.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(width), "n"(CTA_GROUP));
}

// ---- Cluster Launch Control (CLC) for SM100+ ----

// Issue async try_cancel — attempts to cancel a not-yet-launched cluster.
// Response (16 bytes) is written via multicast to [response_addr] in ALL CTAs.
// Completion is signaled via complete_tx on [mbar_addr] in ALL CTAs.
__device__ __forceinline__
void clc_try_cancel(int response_addr, int mbar_addr)
{
    asm volatile(
        "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%0], [%1];"
        :: "r"(response_addr), "r"(mbar_addr)
        : "memory"
    );
}

// Load the 16-byte CLC response from shared memory as 4x32-bit
__device__ __forceinline__
void clc_load_response(int response_addr, uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3)
{
    asm volatile(
        "ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(response_addr)
        : "memory"
    );
}

// Check if the try_cancel succeeded
__device__ __forceinline__
uint32_t clc_is_canceled(uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3)
{
    uint32_t result = 0;
    asm volatile(
        "{\n\t"
        ".reg .b128 handle;\n\t"
        ".reg .pred p;\n\t"
        "mov.b128 handle, {%1, %2, %3, %4};\n\t"
        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, handle;\n\t"
        "@p mov.s32 %0, 1;\n\t"
        "}"
        : "+r"(result)
        : "r"(r0), "r"(r1), "r"(r2), "r"(r3)
    );
    return result;
}

// Extract the x-coordinate of the canceled cluster's first CTA
__device__ __forceinline__
uint32_t clc_get_first_ctaid_x(uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3)
{
    uint32_t x;
    asm volatile(
        "{\n\t"
        ".reg .b128 handle;\n\t"
        "mov.b128 handle, {%1, %2, %3, %4};\n\t"
        "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, handle;\n\t"
        "}"
        : "=r"(x)
        : "r"(r0), "r"(r1), "r"(r2), "r"(r3)
    );
    return x;
}

// Fused load + both queries: one ld.shared.b128 feeds both query_cancel ops
// via a shared .b128 handle register. Returns is_valid (0/1) and new_ctaid
// (only meaningful when is_valid == 1).
__device__ __forceinline__
void clc_query_response(int response_addr, uint32_t &is_valid, uint32_t &new_ctaid)
{
    asm volatile(
        "{\n\t"
        ".reg .b128 handle;\n\t"
        ".reg .pred p;\n\t"
        "mov.s32 %0, 0;\n\t"
        "ld.shared.b128 handle, [%2];\n\t"
        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, handle;\n\t"
        "@p mov.s32 %0, 1;\n\t"
        "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %1, handle;\n\t"
        "}"
        : "=r"(is_valid), "=r"(new_ctaid)
        : "r"(response_addr)
        : "memory"
    );
}

// Cluster-scoped arrive with expected TX count — returns state for try_wait
__device__ __forceinline__
uint64_t mbarrier_arrive_expect_tx_cluster(int mbar_addr, uint32_t tx_count)
{
    uint64_t state;
    asm volatile(
        "mbarrier.arrive.expect_tx.cluster.relaxed.shared::cta.b64 %0, [%1], %2;"
        : "=l"(state)
        : "r"(mbar_addr), "r"(tx_count)
        : "memory"
    );
    return state;
}

// Cluster-scoped mbarrier wait using state from arrive_expect_tx
__device__ __forceinline__
void mbarrier_wait_cluster(int mbar_addr, uint64_t state)
{
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "CLC_WAIT:\n\t"
        "mbarrier.try_wait.cluster.acquire.shared::cta.b64 p, [%0], %1;\n\t"
        "@!p bra CLC_WAIT;\n\t"
        "}"
        :: "r"(mbar_addr), "l"(state)
        : "memory"
    );
}

// CLC fences: release before next iteration's async write, acquire after
__device__ __forceinline__
void clc_fence_release()
{
    asm volatile("fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;" ::: "memory");
}

__device__ __forceinline__
void clc_fence_acquire()
{
    asm volatile("fence.proxy.async::generic.acquire.sync_restrict::shared::cluster.cluster;" ::: "memory");
}

// Relaxed cluster barrier (no acquire/release semantics)
__device__ __forceinline__
void cluster_arrive_relaxed()
{
    asm volatile("barrier.cluster.arrive.relaxed;" ::: "memory");
}

__device__ __forceinline__
void cluster_wait_relaxed()
{
    asm volatile("barrier.cluster.wait;" ::: "memory");
}