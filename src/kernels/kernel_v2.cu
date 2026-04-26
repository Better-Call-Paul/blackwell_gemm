#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_bf16.h>
#include <torch/library.h>

constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 4;
constexpr int SWIZZLE_WIDTH = 64;
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

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
}

inline void check_cu(CUresult err)
{
    if (err == CUDA_SUCCESS) return;
    const char *msg;
    if (cuGetErrorString(err, &msg) != CUDA_SUCCESS) msg = "unable to get error string";
    TORCH_CHECK(false, msg);
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

template<int CTA_GROUP = 1>
__device__ __forceinline__
void dealloc_tmem(int taddr, int width)
{
    asm volatile("tcgen05.dealloc.cta_group::%2.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(width), "n"(CTA_GROUP));
}

constexpr int MMA_K = 16;
constexpr int CTA_GROUP_SIZE = 2;

template<int BM, int BN, int BK>
__global__ __cluster_dims__(2, 1, 1)
void gemm_2sm_mma(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K,
                const __grid_constant__ CUtensorMap A_tmap, const __grid_constant__ CUtensorMap B_tmap)
{
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int cta_rank = get_cluster_cta_rank();

    const int grid_n = N / BN;
    const int group_id = blockIdx.x / CTA_GROUP_SIZE;
    const int block_col = group_id % grid_n;
    const int base_block_row = (group_id / grid_n) * CTA_GROUP_SIZE;
    const int block_row = base_block_row + (blockIdx.x % CTA_GROUP_SIZE);

    extern __shared__ __align__(1024) char smem[];
    const int A_smem = static_cast<int>(__cvta_generic_to_shared(smem));
    const int B_smem = A_smem + (BM * BK) * sizeof(nv_bfloat16);

    #pragma nv_diag_suppress static_var_with_dynamic_init

    __shared__ __align__(8) uint64_t mbar_storage[2];
    int tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(&mbar_storage[0]));
    int mma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(&mbar_storage[1]));

    __shared__ int tmem_smem[1];
    int tmem_smem_addr = static_cast<int>(__cvta_generic_to_shared(tmem_smem));

    if (warp_id == 0 && elect_sync())
    {
        mbarrier_init(tma_mbar_addr, 2);
        mbarrier_init(mma_mbar_addr, 1);
        cluster_fence_mbarrier_init();
    }
    else if (warp_id == 1)
    {
        alloc_tmem<2>(tmem_smem_addr, BN);
    }
    cluster_sync();

    int tma_phase = 0;
    int mma_phase = 0;
    const int tmem_addr = tmem_smem[0];
    constexpr uint32_t i_desc = (1U << 4U)
                              | (1U << 7U)
                              | (1U << 10U)
                              | ((uint32_t)BN >> 3U << 17U)
                              | ((uint32_t)(CTA_GROUP_SIZE * BM) >> 4U << 24U)
                              ;

    int tma_mbar = tma_mbar_addr;
    if (cta_rank == 1)
    {
        tma_mbar = map_smem_addr_to_cta_rank(tma_mbar_addr, 0);
    }

    const int num_iters = K / BK;

    for (int iter_k = 0; iter_k < num_iters; ++iter_k)
    {
        if (warp_id == 0 && elect_sync())
        {
            constexpr int copy_size = (BM + BN / CTA_GROUP_SIZE) * BK * sizeof(nv_bfloat16);
            mbarrier_arrive_expect_cluster(tma_mbar, copy_size);

            int a_row = block_row * BM;
            int b_row = block_col * BN + cta_rank * (BN / CTA_GROUP_SIZE);

            tma_3d_gmem2smem<2>(A_smem, &A_tmap, 0, a_row, iter_k * BK / SWIZZLE_WIDTH, tma_mbar);
            tma_3d_gmem2smem<2>(B_smem, &B_tmap, 0, b_row, iter_k * BK / SWIZZLE_WIDTH, tma_mbar);
        }

        if (cta_rank == 0)
        {
            mbarrier_wait(tma_mbar_addr, tma_phase);
            tcgen05_sync();
            tma_phase ^= 1;
        }

        if (cta_rank == 0 && warp_id == 0 && elect_sync())
        {
            for (int k1 = 0; k1 < BK / SWIZZLE_WIDTH; ++k1)
            {
                for (int k2 = 0; k2 < SWIZZLE_WIDTH / MMA_K; ++k2)
                {
                    const int a_off = k1 * BM * SWIZZLE_WIDTH * sizeof(nv_bfloat16) + k2 * MMA_K * sizeof(nv_bfloat16);
                    const int b_off = k1 * (BN / CTA_GROUP_SIZE) * SWIZZLE_WIDTH * sizeof(nv_bfloat16) + k2 * MMA_K * sizeof(nv_bfloat16);

                    int enable = (iter_k == 0 && k1 == 0 && k2 == 0) ? 0 : 1;
                    uint64_t a_desc = make_smem_desc(A_smem + a_off);
                    uint64_t b_desc = make_smem_desc(B_smem + b_off);

                    tcgen05_mma_bf16<2>(tmem_addr, a_desc, b_desc, i_desc, enable);
                }
            }

            tcgen05_commit_multicast(mma_mbar_addr, 0b11);
        }
        mbarrier_wait(mma_mbar_addr, mma_phase);
        mma_phase ^= 1;
    }

    tcgen05_sync();

    for (int n = 0; n < BN / 8; ++n)
    {
        float tmp[8];
        const int addr = tmem_addr + ((cta_rank * 128 + warp_id * 32) << 16) + n * 8;

        tcgen05_ld(tmp, addr);
        tcgen05_wait_ld();

        nv_bfloat162 out[4];
        for (int i = 0; i < 4; ++i)
        {
            out[i] = __float22bfloat162_rn({tmp[i * 2], tmp[i * 2 + 1]});
        }

        nv_bfloat16 *output_ptr = C + (block_row * BM + tid) * N + (block_col * BN + n * 8);
        reinterpret_cast<int4 *>(output_ptr)[0] = reinterpret_cast<int4 *>(out)[0];
    }

    __syncthreads();

    if (warp_id == 0)
    {
        dealloc_tmem<2>(tmem_addr, BN);
    }
}

template<int BM, int BN, int BK>
void launch_matmul_v2(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K)
{
    CUtensorMap a_tmap, b_tmap;

    init_3d_tma_map(&a_tmap, A, K, BK, M, BM, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B);
    init_3d_tma_map(&b_tmap, B, K, BK, N, BN / CTA_GROUP_SIZE, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B);

    int grid_n = N / BN;
    int grid_m = M / (CTA_GROUP_SIZE * BM);
    dim3 grid(grid_n * grid_m * CTA_GROUP_SIZE);
    int block = TB_SIZE;

    const int smem_size = (BM + BN / CTA_GROUP_SIZE) * BK * sizeof(nv_bfloat16);

    cudaFuncSetAttribute(gemm_2sm_mma<BM, BN, BK>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    gemm_2sm_mma<BM, BN, BK><<<grid, block, smem_size>>>(A, B, C, M, N, K, a_tmap, b_tmap);
}

void matmul_v2(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K)
{
    launch_matmul_v2<128, 256, 128>(A, B, C, M, N, K);
}
