#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_bf16.h>
#include <torch/library.h>
#include "../profiler.h"

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

template<int CTA_GROUP = 1>
__device__ __forceinline__
void dealloc_tmem(int taddr, int width)
{
    asm volatile("tcgen05.dealloc.cta_group::%2.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(width), "n"(CTA_GROUP));
}

constexpr int SM_COUNT = 148;
constexpr int MMA_K = 16;
constexpr int CTA_GROUP_SIZE = 2;
constexpr uint16_t cta_mask = 0b11;
constexpr int NUM_EPILOGUE_WARPS = 4;
constexpr int NUM_PRODUCER_WARPS = 1;
constexpr int NUM_CONSUMER_WARPS = 1;
constexpr int NUM_EPILOGUE_STAGES = 2;
constexpr int TOTAL_WARPS = NUM_EPILOGUE_WARPS + NUM_PRODUCER_WARPS + NUM_CONSUMER_WARPS;

template<int BM, int BN, int BK, int QUEUE_SIZE, bool DO_PROFILE = false>
__global__ __cluster_dims__(2, 1, 1)
void persistent_gemm(const nv_bfloat16* __restrict__ A, const nv_bfloat16* __restrict__ B, nv_bfloat16* __restrict__ C, int M, int N, int K,
                     const __grid_constant__ CUtensorMap A_tmap, const __grid_constant__ CUtensorMap B_tmap,
                     int64_t *profiler_ptr, int num_entries)
{
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;

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

    const int full_mbar_addr = static_cast<int>(__cvta_generic_to_shared(full_mbar_storage));
    const int empty_mbar_addr = static_cast<int>(__cvta_generic_to_shared(empty_mbar_storage));
    const int tmem_empty_addr = static_cast<int>(__cvta_generic_to_shared(tmem_empty));
    const int tmem_full_addr = static_cast<int>(__cvta_generic_to_shared(tmem_full));

    constexpr int producer_warp_id = 4;
    constexpr int consumer_warp_id = 5;

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

    constexpr int copy_size = (BM + BN / CTA_GROUP_SIZE) * BK * sizeof(nv_bfloat16);

    int stage_idx = 0;
    int phase = 0;

    auto advance_stage = [&](int& iter_k)
    {
        ++iter_k;

        stage_idx = (stage_idx + 1) % QUEUE_SIZE;

        phase ^= stage_idx == 0;
    };

    const int k_iters = K / BK;

    int tma_mbar = full_mbar_addr;
    if (cta_rank == 1)
    {
        tma_mbar = map_smem_addr_to_cta_rank(full_mbar_addr, 0);
    }

    const int grid_n = N / BN;
    int wave_iter = -1;
    const int num_clusters = SM_COUNT / CTA_GROUP_SIZE;
    const int total_tiles = (M / BM / CTA_GROUP_SIZE) * (N / BN);
    const int cluster_id = blockIdx.x / CTA_GROUP_SIZE;

    int block_col, block_row;

    auto get_next_tile = [&]()
    {
        int tile_idx = (++wave_iter) * num_clusters + cluster_id;

        if (tile_idx >= total_tiles) return false;

        block_col = tile_idx % grid_n;
        block_row = (tile_idx / grid_n) * CTA_GROUP_SIZE + cta_rank;

        return true;
    };

    if (warp_id == producer_warp_id && elect_sync())
    {
        int global_k = 0;
        while (get_next_tile())
        {
            for (int iter_k = 0; iter_k < k_iters; advance_stage(iter_k))
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
            }
        }
    }
    else if (warp_id == consumer_warp_id && cta_rank == 0 && elect_sync())
    {
        while (get_next_tile())
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

            for (int iter_k = 0; iter_k < k_iters; advance_stage(iter_k))
            {
                int current_full_mbar = full_mbar_addr + stage_idx * 8;
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
            }

            tcgen05_commit_multicast<CTA_GROUP_SIZE>(tmem_full_addr + wave_stage * 8, cta_mask);
        }
    }
    else if (warp_id < NUM_EPILOGUE_WARPS)
    {
        while (get_next_tile())
        {
            int wave_stage = wave_iter % NUM_EPILOGUE_STAGES;
            int wave_phase = (wave_iter / NUM_EPILOGUE_STAGES) & 1;

            if constexpr (DO_PROFILE) if (elect_sync()) profiler.start(ProfilerTag::WaitMainloop);
            mbarrier_wait(tmem_full_addr + wave_stage * 8, wave_phase);
            asm volatile("tcgen05.fence::after_thread_sync;");
            if constexpr (DO_PROFILE) if (elect_sync()) profiler.stop();

            if constexpr (DO_PROFILE) if (elect_sync()) profiler.start(ProfilerTag::Epilogue);
            for (int n = 0; n < BN / 8; ++n)
            {
                float tmp[8];
                const int addr = (tmem_addr + wave_stage * BN) + ((cta_rank * 128 + warp_id * 32) << 16) + n * 8;

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

            const int tmem_empty_mbar = tmem_empty_addr + wave_stage * 8;
            const int tmem_empty_cta0 = tmem_empty_mbar & 0xFEFFFFFF;
            if (elect_sync())
            {
                mbarrier_arrive_cluster(tmem_empty_cta0);
            }
            if constexpr (DO_PROFILE) if (elect_sync()) profiler.stop();
        }

    }

    if constexpr (DO_PROFILE) if (elect_sync()) profiler.flush();

    cluster_sync();

    if (warp_id == 0)
    {
        dealloc_tmem<CTA_GROUP_SIZE>(tmem_addr, BN * NUM_EPILOGUE_STAGES);
    }

}

template<int BM, int BN, int BK, bool DO_PROFILE = false>
void launch_matmul_v5(const nv_bfloat16* A, const nv_bfloat16* B, nv_bfloat16* C, int M, int N, int K,
                      int64_t *profiler_ptr = nullptr, int num_entries = 0)
{
    CUtensorMap A_tmap, B_tmap;
    init_3d_tma_map(&A_tmap, A, K, BK, M, BM, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B);
    init_3d_tma_map(&B_tmap, B, K, BK, N, BN / CTA_GROUP_SIZE, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B);

    constexpr int block_size = (NUM_EPILOGUE_WARPS + NUM_PRODUCER_WARPS + NUM_CONSUMER_WARPS) * WARP_SIZE;

    constexpr int tile_size = (BM + BN / CTA_GROUP_SIZE) * BK * sizeof(nv_bfloat16);
    constexpr int QUEUE_SIZE = 227 * 1024 / tile_size;
    constexpr int smem_size = tile_size * QUEUE_SIZE;

    cudaFuncSetAttribute(persistent_gemm<BM, BN, BK, QUEUE_SIZE, DO_PROFILE>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    dim3 grid(SM_COUNT);

    persistent_gemm<BM, BN, BK, QUEUE_SIZE, DO_PROFILE><<<grid, block_size, smem_size>>>(A, B, C, M, N, K, A_tmap, B_tmap, profiler_ptr, num_entries);
}

void matmul_v5(const nv_bfloat16* A, const nv_bfloat16* B, nv_bfloat16* C, int M, int N, int K)
{
    launch_matmul_v5<128, 256, 64>(A, B, C, M, N, K);
}

void profile_matmul_v5(const nv_bfloat16* A, const nv_bfloat16* B, nv_bfloat16* C, int M, int N, int K,
                        int64_t *profiler_ptr, int num_entries)
{
    launch_matmul_v5<128, 256, 64, true>(A, B, C, M, N, K, profiler_ptr, num_entries);
}
