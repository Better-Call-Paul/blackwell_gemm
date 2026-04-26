#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_bf16.h>
#include <torch/library.h>

constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 4;
constexpr int SWIZZLE_WIDTH = 64;
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

inline void check_cu(CUresult err)
{
    if (err == CUDA_SUCCESS)
    {
        return;
    }
    const char *msg;
    if (cuGetErrorString(err, &msg) != CUDA_SUCCESS)
    {
        msg = "unable to get error string";
    }
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

__device__ __forceinline__
void tma_2d_gmem2smem(int dst, const void *tmap_ptr, int x, int y, int mbar_addr)
{
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(mbar_addr) : "memory");
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
void mbarrier_arrive_expect(const int mbarrier_address, const int copy_size)
{
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                :: "r"(mbarrier_address), "r"(copy_size) : "memory");
}

__device__ __forceinline__
void cluster_fence_mbarrier_init()
{
    asm volatile("fence.mbarrier_init.release.cluster;");
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

constexpr int BM = 128;
constexpr int MMA_K = 16;

template<int BN, int BK, bool TMAP_3d>
__global__ __launch_bounds__(TB_SIZE)
void basic_tcgen05_gemm(const __grid_constant__ CUtensorMap A_tmap,
                    const __grid_constant__ CUtensorMap B_tmap,
                    const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K)
{
    const int col = threadIdx.x;
    const int row = threadIdx.y;

    const int tid = col + row * blockDim.x;
    const int warp_id = tid / 32;

    extern __shared__ __align__(1024) char smem[];
    const int A_smem = static_cast<int>(__cvta_generic_to_shared(smem));
    const int B_smem = A_smem + (BM * BK) * sizeof(nv_bfloat16);

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ __align__(8) uint64_t mbar[1];
    int mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbar));
    __shared__ int tmem[1];
    int tmem_smem_addr = static_cast<int>(__cvta_generic_to_shared(tmem));

    if (warp_id == 0 && elect_sync())
    {
        mbarrier_init(mbar_addr, 1);
        cluster_fence_mbarrier_init();
    }
    else if (warp_id == 1)
    {
        alloc_tmem(tmem_smem_addr, BN);
    }

    __syncthreads();
    const int tmem_addr = tmem[0];

    int phase = 0;
    constexpr uint32_t i_desc = (1U << 4U)
                              | (1U << 7U)
                              | (1U << 10U)
                              | ((uint32_t)BN >> 3U << 17U)
                              | ((uint32_t)BM >> 4U << 24U)
                              ;

    const int num_iters = K / BK;

    for (int iter_k = 0; iter_k < num_iters; ++iter_k)
    {
        if (warp_id == 0 && elect_sync())
        {
            if constexpr (TMAP_3d)
            {
                tma_3d_gmem2smem<1>(A_smem, &A_tmap, 0, blockIdx.y * BM, iter_k * BK / 64, mbar_addr);
                tma_3d_gmem2smem<1>(B_smem, &B_tmap, 0, blockIdx.x * BN, iter_k * BK / 64, mbar_addr);
            }
            else
            {
                for (int k = 0; k < BK / 64; ++k)
                {
                    const int col = iter_k * BK + k * 64;
                    const int offset_a = BM * k * 64 * sizeof(nv_bfloat16);
                    const int offset_b = BN * k * 64 * sizeof(nv_bfloat16);

                    tma_2d_gmem2smem(A_smem + offset_a, &A_tmap, col, blockIdx.y * BM, mbar_addr);
                    tma_2d_gmem2smem(B_smem + offset_b, &B_tmap, col, blockIdx.x * BN, mbar_addr);
                }
            }

            constexpr int copy_size = (BM + BN) * BK * sizeof(nv_bfloat16);
            mbarrier_arrive_expect(mbar_addr, copy_size);
        }

        mbarrier_wait(mbar_addr, phase);
        asm volatile("tcgen05.fence::after_thread_sync;");
        phase ^= 1;

        if (warp_id == 0 && elect_sync())
        {
            tcgen05_mma_bf16<1>(tmem_addr, make_smem_desc(A_smem), make_smem_desc(B_smem), i_desc, iter_k);

            for (int k2 = 1; k2 < 64 / MMA_K; ++k2)
            {
                const int offset = MMA_K * k2 * sizeof(nv_bfloat16);
                const uint64_t a_smem_desc = make_smem_desc(A_smem + offset);
                const uint64_t b_smem_desc = make_smem_desc(B_smem + offset);
                tcgen05_mma_bf16<1>(tmem_addr, a_smem_desc, b_smem_desc, i_desc, 1);
            }


            for (int k1 = 1; k1 < BK / 64; ++k1)
            {
                for (int k2 = 0; k2 < 64 / MMA_K; ++k2)
                {
                    const int offset = k1 * BM * 64 * sizeof(nv_bfloat16) + k2 * MMA_K * sizeof(nv_bfloat16);
                    const uint64_t a_smem_desc = make_smem_desc(A_smem + offset);
                    const int b_offset = k1 * BN * 64 * sizeof(nv_bfloat16) + k2 * MMA_K * sizeof(nv_bfloat16);
                    const uint64_t b_smem_desc = make_smem_desc(B_smem + b_offset);
                    tcgen05_mma_bf16<1>(tmem_addr, a_smem_desc, b_smem_desc, i_desc, 1);
                }
            }

            tcgen05_commit(mbar_addr);
        }

        mbarrier_wait(mbar_addr, phase);
        phase ^= 1;

    }

    tcgen05_sync();

    for (int n = 0; n < BN / 8; ++n)
    {
        float tmp[8];
        const int addr = tmem_addr + ((warp_id * 32) << 16) + n * 8;

        tcgen05_ld(tmp, addr);
        tcgen05_wait_ld();

        nv_bfloat162 out[4];
        for (int i = 0; i < 4; ++i)
        {
            out[i] = __float22bfloat162_rn({tmp[i * 2], tmp[i * 2 + 1]});
        }

        nv_bfloat16 *output_ptr = C + (blockIdx.y * BM + tid) * N + (blockIdx.x * BN + n * 8);
        reinterpret_cast<int4 *>(output_ptr)[0] = reinterpret_cast<int4 *>(out)[0];
    }

    __syncthreads();
    if (warp_id == 0)
    {
        dealloc_tmem(tmem_addr, BN);
    }

}

template<int BN, int BK, bool TMAP_3d>
void launch_matmul_v1(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K)
{
    CUtensorMap a_tmap, b_tmap;

    if (TMAP_3d)
    {
        init_3d_tma_map(&a_tmap, A, K, BK, M, BM, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B);
        init_3d_tma_map(&b_tmap, B, K, BK, N, BN, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B);
    }
    else
    {
        init_tmap_2d(&a_tmap, A, M, K, BM, 64, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B);
        init_tmap_2d(&b_tmap, B, N, K, BN, 64, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B);
    }

    dim3 grid(N / BN, M / BM);
    const int block_size = TB_SIZE;

    const int dynamic_smem_size = (BM * BK + BN * BK) * sizeof(nv_bfloat16);

    auto kernel = basic_tcgen05_gemm<BN, BK, TMAP_3d>;
    if (dynamic_smem_size > 48'000)
    {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_smem_size);
    }

    kernel<<<grid, block_size, dynamic_smem_size>>>(a_tmap, b_tmap, A, B, C, M, N, K);
}


void matmul_v1_2d(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K)
{
    launch_matmul_v1<256, 128, false>(A, B, C, M, N, K);
}

void matmul_v1_3d(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K)
{
    launch_matmul_v1<256, 128, true>(A, B, C, M, N, K);
}
