#include <torch/extension.h>
#include <cuda_bf16.h>

typedef void MatmulFn(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K);

MatmulFn matmul_v1_2d;
MatmulFn matmul_v1_3d;
MatmulFn matmul_v2;
MatmulFn matmul_v3;
MatmulFn matmul_v4;
MatmulFn matmul_v5;
MatmulFn matmul_v6;
MatmulFn matmul_v7;
MatmulFn matmul_v8;

void profile_matmul_v5(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K,
                        int64_t *profiler_ptr, int num_entries);
void profile_matmul_v6(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K,
                        int64_t *profiler_ptr, int num_entries);
void profile_matmul_v7(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K,
                        int64_t *profiler_ptr, int num_entries);
void profile_matmul_v8(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K,
                        int64_t *profiler_ptr, int num_entries);

template<MatmulFn matmul_fn>
at::Tensor matmul(const at::Tensor& A, const at::Tensor& B)
{
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    auto C = at::empty({M, N}, A.options());
    matmul_fn(
        reinterpret_cast<nv_bfloat16 *>(A.data_ptr()),
        reinterpret_cast<nv_bfloat16 *>(B.data_ptr()),
        reinterpret_cast<nv_bfloat16 *>(C.data_ptr()),
        M, N, K
    );
    return C;
}

at::Tensor profile_matmul_v5_wrapper(const at::Tensor& A, const at::Tensor& B, const at::Tensor& profiler, int num_entries)
{
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    auto C = at::empty({M, N}, A.options());
    profile_matmul_v5(
        reinterpret_cast<nv_bfloat16 *>(A.data_ptr()),
        reinterpret_cast<nv_bfloat16 *>(B.data_ptr()),
        reinterpret_cast<nv_bfloat16 *>(C.data_ptr()),
        M, N, K,
        profiler.data_ptr<int64_t>(),
        num_entries
    );
    return C;
}

at::Tensor profile_matmul_v6_wrapper(const at::Tensor& A, const at::Tensor& B, const at::Tensor& profiler, int num_entries)
{
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    auto C = at::empty({M, N}, A.options());
    profile_matmul_v6(
        reinterpret_cast<nv_bfloat16 *>(A.data_ptr()),
        reinterpret_cast<nv_bfloat16 *>(B.data_ptr()),
        reinterpret_cast<nv_bfloat16 *>(C.data_ptr()),
        M, N, K,
        profiler.data_ptr<int64_t>(),
        num_entries
    );
    return C;
}

at::Tensor profile_matmul_v7_wrapper(const at::Tensor& A, const at::Tensor& B, const at::Tensor& profiler, int num_entries)
{
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    auto C = at::empty({M, N}, A.options());
    profile_matmul_v7(
        reinterpret_cast<nv_bfloat16 *>(A.data_ptr()),
        reinterpret_cast<nv_bfloat16 *>(B.data_ptr()),
        reinterpret_cast<nv_bfloat16 *>(C.data_ptr()),
        M, N, K,
        profiler.data_ptr<int64_t>(),
        num_entries
    );
    return C;
}

at::Tensor profile_matmul_v8_wrapper(const at::Tensor& A, const at::Tensor& B, const at::Tensor& profiler, int num_entries)
{
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    auto C = at::empty({M, N}, A.options());
    profile_matmul_v8(
        reinterpret_cast<nv_bfloat16 *>(A.data_ptr()),
        reinterpret_cast<nv_bfloat16 *>(B.data_ptr()),
        reinterpret_cast<nv_bfloat16 *>(C.data_ptr()),
        M, N, K,
        profiler.data_ptr<int64_t>(),
        num_entries
    );
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("matmul_v1_2d", &matmul<matmul_v1_2d>);
    m.def("matmul_v1_3d", &matmul<matmul_v1_3d>);
    m.def("matmul_v2", &matmul<matmul_v2>);
    m.def("matmul_v3", &matmul<matmul_v3>);
    m.def("matmul_v4", &matmul<matmul_v4>);
    m.def("matmul_v5", &matmul<matmul_v5>);
    m.def("profile_matmul_v5", &profile_matmul_v5_wrapper);
    m.def("matmul_v6", &matmul<matmul_v6>);
    m.def("profile_matmul_v6", &profile_matmul_v6_wrapper);
    m.def("matmul_v7", &matmul<matmul_v7>);
    m.def("profile_matmul_v7", &profile_matmul_v7_wrapper);
    m.def("matmul_v8", &matmul<matmul_v8>);
    m.def("profile_matmul_v8", &profile_matmul_v8_wrapper);
}
