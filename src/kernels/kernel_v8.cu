#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_bf16.h>
#include <torch/library.h>
#include "../profiler.h"
#include <cstdlib>
#include <cstring>
#include <vector>

static CUtensorMapL2promotion parse_l2_env(const char *name)
{
    const char *v = std::getenv(name);
    if (!v)
    {
        return CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE;
    }
    if (std::strcmp(v, "256") == 0)
    {
        return CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
    }
    if (std::strcmp(v, "128") == 0)
    {
        return CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    }
    if (std::strcmp(v, "64") == 0)
    {
        return CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_64B;
    }
    return CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE;
}

constexpr int WARP_SIZE = 32;
constexpr int SWIZZLE_WIDTH = 64;

inline void check_cu(CUresult err)
{
    if (err == CUDA_SUCCESS) return;
    const char *msg;
    if (cuGetErrorString(err, &msg) != CUDA_SUCCESS) msg = "unable to get error string";
    TORCH_CHECK(false, msg);
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

__device__ __forceinline__
void mbarrier_wait(int mbar_addr, int phase)
{
    uint32_t ticks = 0x989680;
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

template <int CTA_GROUP = 1>
__device__ __forceinline__
void tma_3d_gmem2smem(int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr)
{
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::%6 "
                "[%0], [%1, {%2, %3, %4}], [%5];"
                :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(mbar_addr), "n"(CTA_GROUP)
                : "memory");
}

__device__ __forceinline__
constexpr uint64_t desc_encode(uint64_t x)
{
    return (x & 0x3'FFFFULL) >> 4ULL;
};

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
    uint64_t globalStrides[rank-1] = {global_width * sizeof(nv_bfloat16)};
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
    uint64_t globalStrides[rank-1] = {(uint64_t)K * sizeof(nv_bfloat16), 128};
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
        swizzle,
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
        "tcgen05.mma.cta_group::%5.kind::f16 [%0], %1, %2, %3, p;\n\t"
        "}"
        :: "r"(tmem_address), "l"(a_descr), "l"(b_descr), "r"(i_descr), "r"(enable_input_d), "n"(CTA_GROUP)
    );
}

__device__ __forceinline__
uint64_t make_smem_desc(int addr)
{
    const int stride_byte_offset = 8 * SWIZZLE_WIDTH * sizeof(__nv_bfloat16);
    return desc_encode(addr) | (desc_encode(stride_byte_offset) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
}

__device__ __forceinline__
void mbarrier_arrive_expect_cluster(const int mbarrier_address, const int copy_size)
{
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                :: "r"(mbarrier_address), "r"(copy_size) : "memory");
}

__device__ __forceinline__
void mbarrier_arrive_cluster(const int mbarrier_address)
{
    asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                :: "r"(mbarrier_address) : "memory");
}

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

template<int CTA_GROUP = 2>
__device__ __forceinline__
void tcgen05_commit_multicast(const int mbarrier_address, uint16_t ctaMask)
{
    asm volatile("tcgen05.commit.cta_group::%2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                        :: "r"(mbarrier_address), "h"(ctaMask), "n"(CTA_GROUP) : "memory");
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

__device__ __forceinline__
void barrier_sync(int bar_id, int thread_count)
{
    asm volatile("bar.sync %0, %1;" :: "r"(bar_id), "r"(thread_count));
}

template<int CTA_GROUP = 1>
__device__ __forceinline__
void dealloc_tmem(int taddr, int width)
{
    asm volatile("tcgen05.dealloc.cta_group::%2.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(width), "n"(CTA_GROUP));
}

__device__ __forceinline__
void clc_try_cancel(int response_addr, int mbar_addr)
{
    asm volatile(
        "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%0], [%1];"
        :: "r"(response_addr), "r"(mbar_addr)
        : "memory"
    );
}

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

constexpr int MMA_K = 16;
constexpr int CTA_GROUP_SIZE = 2;
constexpr uint16_t cta_mask = 0b11;
constexpr int NUM_EPILOGUE_WARPS = 4;
constexpr int NUM_PRODUCER_WARPS = 1;
constexpr int NUM_CONSUMER_WARPS = 1;
constexpr int NUM_SCHEDULER_WARPS = 1;
constexpr int NUM_EPILOGUE_STAGES = 2;
constexpr int TOTAL_WARPS = NUM_EPILOGUE_WARPS + NUM_PRODUCER_WARPS + NUM_CONSUMER_WARPS + NUM_SCHEDULER_WARPS;
constexpr int STORE_N = 64;
constexpr int NUM_TMA_STORE_STAGES = 2;
constexpr int NUM_CLC_STAGES = 2;
constexpr int CLC_EMPTY_ARRIVE = TOTAL_WARPS * CTA_GROUP_SIZE;

static void hilbert_d2xy_host(int n, int d, int &x, int &y)
{
    x = y = 0;
    for (int s = 1; s < n; s <<= 1)
    {
        int rx = 1 & (d >> 1);
        int ry = 1 & (d ^ rx);
        if (ry == 0)
        {
            if (rx == 1)
            {
                x = s - 1 - x;
                y = s - 1 - y;
            }
            int t = x; x = y; y = t;
        }
        x += s * rx;
        y += s * ry;
        d >>= 2;
    }
}

template<int BM, int BN, int BK, int QUEUE_SIZE, bool DO_PROFILE = false>
__global__ __cluster_dims__(2, 1, 1)
void clc_hilbert_gemm(
    const nv_bfloat16* __restrict__ A,
    const nv_bfloat16* __restrict__ B,
    nv_bfloat16* __restrict__ C,
    int M, int N, int K,
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B_tmap,
    const __grid_constant__ CUtensorMap C_tmap,
    const int* __restrict__ tile_map,
    int64_t *profiler_ptr,
    int num_entries)
{
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const uint32_t cta_rank = get_cluster_cta_rank();

    extern __shared__ __align__(1024) char smem[];
    int smem_ptr = static_cast<int>(__cvta_generic_to_shared(smem));

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ int tmem[1];
    int tmem_smem_addr = static_cast<int>(__cvta_generic_to_shared(tmem));

    __shared__ __align__(8) uint64_t full_mbar_storage[QUEUE_SIZE];
    __shared__ __align__(8) uint64_t empty_mbar_storage[QUEUE_SIZE];
    __shared__ __align__(8) uint64_t tmem_full[NUM_EPILOGUE_STAGES];
    __shared__ __align__(8) uint64_t tmem_empty[NUM_EPILOGUE_STAGES];

    __shared__ __align__(16) uint8_t  clc_response[NUM_CLC_STAGES][16];
    __shared__ __align__(8)  uint64_t clc_full_mbar[NUM_CLC_STAGES];
    __shared__ __align__(8)  uint64_t clc_empty_mbar[NUM_CLC_STAGES];

    const int full_mbar_addr      = static_cast<int>(__cvta_generic_to_shared(full_mbar_storage));
    const int empty_mbar_addr     = static_cast<int>(__cvta_generic_to_shared(empty_mbar_storage));
    const int tmem_empty_addr     = static_cast<int>(__cvta_generic_to_shared(tmem_empty));
    const int tmem_full_addr      = static_cast<int>(__cvta_generic_to_shared(tmem_full));
    const int clc_response_addr   = static_cast<int>(__cvta_generic_to_shared(clc_response));
    const int clc_full_mbar_addr  = static_cast<int>(__cvta_generic_to_shared(clc_full_mbar));
    const int clc_empty_mbar_addr = static_cast<int>(__cvta_generic_to_shared(clc_empty_mbar));

    constexpr int producer_warp_id  = 4;
    constexpr int consumer_warp_id  = 5;
    constexpr int scheduler_warp_id = 6;

    constexpr int copy_size = (BM + BN / CTA_GROUP_SIZE) * BK * sizeof(nv_bfloat16);
    constexpr int store_buf_size = BM * STORE_N * sizeof(nv_bfloat16);
    const int store_smem = smem_ptr + QUEUE_SIZE * copy_size;
    nv_bfloat16 *store_base = reinterpret_cast<nv_bfloat16*>(smem + QUEUE_SIZE * copy_size);

    Profiler profiler;
    if constexpr (DO_PROFILE) if (elect_sync())
    {
        profiler.init(num_entries, profiler_ptr, blockIdx.x * TOTAL_WARPS + warp_id);
        profiler.start(ProfilerTag::Setup);
    }

    if (warp_id == producer_warp_id && elect_sync())
    {
        for (int i = 0; i < QUEUE_SIZE; ++i)
        {
            mbarrier_init(full_mbar_addr + i * 8, NUM_PRODUCER_WARPS * CTA_GROUP_SIZE);
            mbarrier_init(empty_mbar_addr + i * 8, 1);
        }
        for (int i = 0; i < NUM_EPILOGUE_STAGES; ++i)
        {
            mbarrier_init(tmem_full_addr + i * 8, 1);
            mbarrier_init(tmem_empty_addr + i * 8, NUM_EPILOGUE_WARPS * CTA_GROUP_SIZE);
        }
        for (int i = 0; i < NUM_CLC_STAGES; ++i)
        {
            mbarrier_init(clc_full_mbar_addr + i * 8, 1);
            mbarrier_init(clc_empty_mbar_addr + i * 8, CLC_EMPTY_ARRIVE);
        }

        cluster_fence_mbarrier_init();
    }
    else if (warp_id == consumer_warp_id)
    {
        alloc_tmem<CTA_GROUP_SIZE>(tmem_smem_addr, BN * NUM_EPILOGUE_STAGES);
    }

    cluster_sync();

    if constexpr (DO_PROFILE) if (elect_sync()) profiler.stop();

    const int tmem_addr = tmem[0];
    constexpr uint32_t i_desc = (1U << 4U)
                              | (1U << 7U)
                              | (1U << 10U)
                              | ((uint32_t)BN >> 3U << 17U)
                              | ((uint32_t)(CTA_GROUP_SIZE * BM) >> 4U << 24U)
                              ;

    const int k_iters = K / BK;

    int tma_mbar = full_mbar_addr;
    if (cta_rank == 1)
    {
        tma_mbar = map_smem_addr_to_cta_rank(full_mbar_addr, 0);
    }

    const int cta0_clc_empty_addr = (cta_rank == 0)
        ? clc_empty_mbar_addr
        : map_smem_addr_to_cta_rank(clc_empty_mbar_addr, 0);

    if (warp_id == producer_warp_id && elect_sync())
    {
        int stage_idx = 0;
        int phase = 0;
        int global_k = 0;
        int clc_stage = 0;
        int clc_full_phase = 0;

        int packed = tile_map[blockIdx.x / CTA_GROUP_SIZE];
        int block_col = packed & 0xFFFF;
        int block_row = (packed >> 16) * CTA_GROUP_SIZE + cta_rank;

        while (true)
        {
            for (int iter_k = 0; iter_k < k_iters; ++iter_k)
            {
                if (global_k >= QUEUE_SIZE)
                {
                    if constexpr (DO_PROFILE) profiler.start(ProfilerTag::WaitMMA);
                    mbarrier_wait(empty_mbar_addr + stage_idx * 8, phase ^ 1);
                    if constexpr (DO_PROFILE) profiler.stop();
                }
                ++global_k;

                if constexpr (DO_PROFILE) profiler.start(ProfilerTag::IssueTMA);
                int A_smem_s = smem_ptr + stage_idx * copy_size;
                int B_smem_s = A_smem_s + BM * BK * sizeof(nv_bfloat16);

                int A_row = block_row * BM;
                int B_col = block_col * BN + cta_rank * (BN / CTA_GROUP_SIZE);

                int local_full_mbar = tma_mbar + stage_idx * 8;
                mbarrier_arrive_expect_cluster(local_full_mbar, copy_size);

                tma_3d_gmem2smem<CTA_GROUP_SIZE>(A_smem_s, &A_tmap, 0, A_row, iter_k * BK / SWIZZLE_WIDTH, local_full_mbar);
                tma_3d_gmem2smem<CTA_GROUP_SIZE>(B_smem_s, &B_tmap, 0, B_col, iter_k * BK / SWIZZLE_WIDTH, local_full_mbar);
                if constexpr (DO_PROFILE) profiler.stop();

                stage_idx = (stage_idx + 1) % QUEUE_SIZE;
                phase ^= stage_idx == 0;
            }

            if constexpr (DO_PROFILE) profiler.start(ProfilerTag::WaitMainloop);
            mbarrier_wait(clc_full_mbar_addr + clc_stage * 8, clc_full_phase);
            if constexpr (DO_PROFILE) profiler.stop();

            uint32_t is_valid, new_ctaid;
            clc_query_response(clc_response_addr + clc_stage * 16, is_valid, new_ctaid);

            mbarrier_arrive_cluster(cta0_clc_empty_addr + clc_stage * 8);

            clc_stage = (clc_stage + 1) % NUM_CLC_STAGES;
            if (clc_stage == 0) clc_full_phase ^= 1;

            if (!is_valid) break;

            packed = tile_map[new_ctaid / CTA_GROUP_SIZE];
            block_col = packed & 0xFFFF;
            block_row = (packed >> 16) * CTA_GROUP_SIZE + cta_rank;
        }
    }
    else if (warp_id == consumer_warp_id && elect_sync())
    {
        int stage_idx = 0;
        int phase = 0;
        int wave_iter = 0;
        int clc_stage = 0;
        int clc_full_phase = 0;

        while (true)
        {
            if (cta_rank == 0)
            {
                int wave_stage = wave_iter % NUM_EPILOGUE_STAGES;
                int wave_phase = (wave_iter / NUM_EPILOGUE_STAGES) & 1;

                if (wave_iter >= NUM_EPILOGUE_STAGES)
                {
                    if constexpr (DO_PROFILE) profiler.start(ProfilerTag::WaitEpilogue);
                    mbarrier_wait(tmem_empty_addr + wave_stage * 8, wave_phase ^ 1);
                    if constexpr (DO_PROFILE) profiler.stop();
                }

                asm volatile("tcgen05.fence::after_thread_sync;");

                for (int iter_k = 0; iter_k < k_iters; ++iter_k)
                {
                    int current_full_mbar  = full_mbar_addr  + stage_idx * 8;
                    int current_empty_mbar = empty_mbar_addr + stage_idx * 8;

                    if constexpr (DO_PROFILE) profiler.start(ProfilerTag::WaitTMA);
                    mbarrier_wait(current_full_mbar, phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if constexpr (DO_PROFILE) profiler.stop();

                    if constexpr (DO_PROFILE) profiler.start(ProfilerTag::IssueMMA);
                    int A_smem_s = smem_ptr + stage_idx * copy_size;
                    int B_smem_s = A_smem_s + BM * BK * sizeof(nv_bfloat16);

                    for (int k1 = 0; k1 < BK / SWIZZLE_WIDTH; ++k1)
                    {
                        for (int k2 = 0; k2 < SWIZZLE_WIDTH / MMA_K; ++k2)
                        {
                            const int enable_accum = (iter_k == 0 && k1 == 0 && k2 == 0) ? 0 : 1;

                            const int a_offset = k1 * SWIZZLE_WIDTH * BM * sizeof(nv_bfloat16) + k2 * MMA_K * sizeof(nv_bfloat16);
                            const int b_offset = k1 * SWIZZLE_WIDTH * (BN / CTA_GROUP_SIZE) * sizeof(nv_bfloat16) + k2 * MMA_K * sizeof(nv_bfloat16);

                            uint64_t a_desc = make_smem_desc(A_smem_s + a_offset);
                            uint64_t b_desc = make_smem_desc(B_smem_s + b_offset);

                            tcgen05_mma_bf16<CTA_GROUP_SIZE>(tmem_addr + wave_stage * BN, a_desc, b_desc, i_desc, enable_accum);
                        }
                    }

                    tcgen05_commit_multicast<CTA_GROUP_SIZE>(current_empty_mbar, cta_mask);
                    if constexpr (DO_PROFILE) profiler.stop();

                    stage_idx = (stage_idx + 1) % QUEUE_SIZE;
                    phase ^= stage_idx == 0;
                }

                tcgen05_commit_multicast<CTA_GROUP_SIZE>(tmem_full_addr + wave_stage * 8, cta_mask);
                wave_iter++;
            }

            mbarrier_wait(clc_full_mbar_addr + clc_stage * 8, clc_full_phase);

            uint32_t is_valid, new_ctaid;
            clc_query_response(clc_response_addr + clc_stage * 16, is_valid, new_ctaid);

            mbarrier_arrive_cluster(cta0_clc_empty_addr + clc_stage * 8);

            clc_stage = (clc_stage + 1) % NUM_CLC_STAGES;
            if (clc_stage == 0) clc_full_phase ^= 1;

            if (!is_valid) break;
        }
    }
    else if (warp_id == scheduler_warp_id && elect_sync())
    {
        if (cta_rank == 0)
        {
            #pragma unroll
            for (int s = 0; s < NUM_CLC_STAGES; ++s)
            {
                mbarrier_arrive_expect_cluster(clc_full_mbar_addr + s * 8, 16);
                int remote_full = map_smem_addr_to_cta_rank(clc_full_mbar_addr + s * 8, 1);
                mbarrier_arrive_expect_cluster(remote_full, 16);
                clc_try_cancel(clc_response_addr + s * 16, clc_full_mbar_addr + s * 8);
            }
        }

        int clc_stage = 0;
        int clc_full_phase = 0;
        int clc_empty_phase = 0;

        while (true)
        {
            mbarrier_wait(clc_full_mbar_addr + clc_stage * 8, clc_full_phase);

            uint32_t is_valid, new_ctaid;
            clc_query_response(clc_response_addr + clc_stage * 16, is_valid, new_ctaid);

            mbarrier_arrive_cluster(cta0_clc_empty_addr + clc_stage * 8);

            int issue_stage = clc_stage;
            clc_stage = (clc_stage + 1) % NUM_CLC_STAGES;
            if (clc_stage == 0) clc_full_phase ^= 1;

            if (!is_valid) break;

            if (cta_rank == 0)
            {
                mbarrier_wait(clc_empty_mbar_addr + issue_stage * 8, clc_empty_phase);
                if (issue_stage == NUM_CLC_STAGES - 1) clc_empty_phase ^= 1;

                mbarrier_arrive_expect_cluster(clc_full_mbar_addr + issue_stage * 8, 16);
                int remote_full = map_smem_addr_to_cta_rank(clc_full_mbar_addr + issue_stage * 8, 1);
                mbarrier_arrive_expect_cluster(remote_full, 16);

                clc_try_cancel(clc_response_addr + issue_stage * 16, clc_full_mbar_addr + issue_stage * 8);
            }
        }
    }
    else if (warp_id < NUM_EPILOGUE_WARPS)
    {
        constexpr int EPILOGUE_BAR = 7;
        constexpr int EPILOGUE_THREADS = NUM_EPILOGUE_WARPS * WARP_SIZE;
        constexpr int num_chunks = BN / STORE_N;
        constexpr int LOADS_PER_CHUNK = STORE_N / 8;

        int wave_iter = 0;
        int clc_stage = 0;
        int clc_full_phase = 0;

        int packed = tile_map[blockIdx.x / CTA_GROUP_SIZE];
        int block_col = packed & 0xFFFF;
        int block_row = (packed >> 16) * CTA_GROUP_SIZE + cta_rank;

        while (true)
        {
            int wave_stage = wave_iter % NUM_EPILOGUE_STAGES;
            int wave_phase = (wave_iter / NUM_EPILOGUE_STAGES) & 1;

            if constexpr (DO_PROFILE) if (elect_sync()) profiler.start(ProfilerTag::WaitMainloop);
            mbarrier_wait(tmem_full_addr + wave_stage * 8, wave_phase);
            asm volatile("tcgen05.fence::after_thread_sync;");
            if constexpr (DO_PROFILE) if (elect_sync()) profiler.stop();

            if constexpr (DO_PROFILE) if (elect_sync()) profiler.start(ProfilerTag::Epilogue);

            const int tmem_row = (tmem_addr + wave_stage * BN) + ((cta_rank * 128 + warp_id * 32) << 16);
            const int row = warp_id * 32 + lane_id;
            int store_stage = 0;

            for (int chunk = 0; chunk < num_chunks; ++chunk)
            {
                if (warp_id == 0) tma_store_wait<NUM_TMA_STORE_STAGES - 1>();

                float tmp[LOADS_PER_CHUNK][8];
                #pragma unroll
                for (int n = 0; n < LOADS_PER_CHUNK; ++n)
                {
                    tcgen05_ld(tmp[n], tmem_row + chunk * STORE_N + n * 8);
                }
                tcgen05_wait_ld();

                if (chunk == num_chunks - 1)
                {
                    tcgen05_before_thread_sync();
                    const int tmem_empty_cta0 = (tmem_empty_addr + wave_stage * 8) & 0xFEFFFFFF;
                    if (elect_sync())
                    {
                        mbarrier_arrive_cluster(tmem_empty_cta0);
                    }
                }

                barrier_sync(EPILOGUE_BAR, EPILOGUE_THREADS);

                #pragma unroll
                for (int n = 0; n < LOADS_PER_CHUNK; ++n)
                {
                    nv_bfloat162 packed_bf[4];
                    #pragma unroll
                    for (int i = 0; i < 4; ++i)
                    {
                        packed_bf[i] = __float22bfloat162_rn({tmp[n][i * 2], tmp[n][i * 2 + 1]});
                    }
                    const int swizzled_n = n ^ (row & 7);
                    nv_bfloat16 *write_ptr = store_base + store_stage * BM * STORE_N + row * STORE_N + swizzled_n * 8;
                    *reinterpret_cast<int4*>(write_ptr) = *reinterpret_cast<int4*>(packed_bf);
                }

                __syncwarp();
                tma_store_fence();
                barrier_sync(EPILOGUE_BAR, EPILOGUE_THREADS);

                if (warp_id == 0 && elect_sync())
                {
                    const int src = store_smem + store_stage * store_buf_size;
                    tma_2d_smem2gmem(src, &C_tmap, block_col * BN + chunk * STORE_N, block_row * BM);
                    tma_store_commit();
                }

                store_stage ^= 1;
            }

            if constexpr (DO_PROFILE) if (elect_sync()) profiler.stop();

            wave_iter++;

            mbarrier_wait(clc_full_mbar_addr + clc_stage * 8, clc_full_phase);

            uint32_t is_valid, new_ctaid;
            clc_query_response(clc_response_addr + clc_stage * 16, is_valid, new_ctaid);

            if (elect_sync())
            {
                mbarrier_arrive_cluster(cta0_clc_empty_addr + clc_stage * 8);
            }

            clc_stage = (clc_stage + 1) % NUM_CLC_STAGES;
            if (clc_stage == 0) clc_full_phase ^= 1;

            if (!is_valid) break;

            packed = tile_map[new_ctaid / CTA_GROUP_SIZE];
            block_col = packed & 0xFFFF;
            block_row = (packed >> 16) * CTA_GROUP_SIZE + cta_rank;
        }

        if (warp_id == 0) tma_store_wait<0>();
        barrier_sync(EPILOGUE_BAR, EPILOGUE_THREADS);
    }

    if constexpr (DO_PROFILE) if (elect_sync()) profiler.flush();

    cluster_sync();

    if (warp_id == 0)
    {
        dealloc_tmem<CTA_GROUP_SIZE>(tmem_addr, BN * NUM_EPILOGUE_STAGES);
    }
}

template<int BM, int BN, int BK, bool DO_PROFILE = false>
void launch_matmul_v8(const nv_bfloat16* A, const nv_bfloat16* B, nv_bfloat16* C, int M, int N, int K,
                      int64_t *profiler_ptr = nullptr, int num_entries = 0)
{
    CUtensorMap A_tmap, B_tmap, C_tmap;
    const CUtensorMapL2promotion l2_A = parse_l2_env("GEMM_L2_A");
    const CUtensorMapL2promotion l2_B = parse_l2_env("GEMM_L2_B");
    init_3d_tma_map(&A_tmap, A, K, BK, M, BM, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B, l2_A);
    init_3d_tma_map(&B_tmap, B, K, BK, N, BN / CTA_GROUP_SIZE, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B, l2_B);
    init_tmap_2d(&C_tmap, C, M, N, BM, STORE_N, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B);

    constexpr int block_size = TOTAL_WARPS * WARP_SIZE;
    constexpr int tile_size = (BM + BN / CTA_GROUP_SIZE) * BK * sizeof(nv_bfloat16);
    constexpr int store_total = NUM_TMA_STORE_STAGES * BM * STORE_N * sizeof(nv_bfloat16);
    constexpr int QUEUE_SIZE = (227 * 1024 - store_total) / tile_size;
    constexpr int smem_size = QUEUE_SIZE * tile_size + store_total;

    const int grid_m = M / BM / CTA_GROUP_SIZE;
    const int grid_n = N / BN;
    const int total_tiles = grid_m * grid_n;

    static int* d_tile_map = nullptr;
    static int cached_M = 0, cached_N = 0;

    if (M != cached_M || N != cached_N)
    {
        int hilbert_n = 1;
        while (hilbert_n < grid_m || hilbert_n < grid_n) hilbert_n <<= 1;

        std::vector<int> tile_map_host(total_tiles);
        int idx = 0;
        for (int d = 0; d < hilbert_n * hilbert_n; d++)
        {
            int hx, hy;
            hilbert_d2xy_host(hilbert_n, d, hx, hy);
            if (hx >= grid_n || hy >= grid_m) continue;
            tile_map_host[idx++] = (hy << 16) | hx;
        }

        if (d_tile_map) cudaFree(d_tile_map);
        cudaMalloc(&d_tile_map, total_tiles * sizeof(int));
        cudaMemcpy(d_tile_map, tile_map_host.data(), total_tiles * sizeof(int), cudaMemcpyHostToDevice);

        cached_M = M;
        cached_N = N;
    }

    auto kernel = clc_hilbert_gemm<BM, BN, BK, QUEUE_SIZE, DO_PROFILE>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    dim3 grid(total_tiles * CTA_GROUP_SIZE);
    kernel<<<grid, block_size, smem_size>>>(A, B, C, M, N, K, A_tmap, B_tmap, C_tmap,
                                            d_tile_map, profiler_ptr, num_entries);
}

void matmul_v8(const nv_bfloat16* A, const nv_bfloat16* B, nv_bfloat16* C, int M, int N, int K)
{
    launch_matmul_v8<128, 256, 64>(A, B, C, M, N, K);
}

void profile_matmul_v8(const nv_bfloat16* A, const nv_bfloat16* B, nv_bfloat16* C, int M, int N, int K,
                        int64_t *profiler_ptr, int num_entries)
{
    launch_matmul_v8<128, 256, 64, true>(A, B, C, M, N, K, profiler_ptr, num_entries);
}
