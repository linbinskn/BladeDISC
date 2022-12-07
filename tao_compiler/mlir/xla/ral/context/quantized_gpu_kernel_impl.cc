#include <cublasLt.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <functional>
#include <iostream>

#include "absl/types/optional.h"
#include "tensorflow/compiler/mlir/xla/ral/context/common_context_impl.h"
#include "tensorflow/compiler/mlir/xla/ral/context/context_util.h"
#include "tensorflow/compiler/mlir/xla/ral/context/stream_executor_based_impl.h"
#include "tensorflow/compiler/mlir/xla/ral/device/gpu/gpu_driver.h"

#if defined(PLATFORM_ALIBABA) and defined(ENABLE_BLADE_GEMM)
#include "bladnn/bladnn.h"
#endif

namespace tao {
namespace ral {

namespace gpu {

namespace se = ::stream_executor;

namespace se_impl {

struct CublasLtBiasParamsKey {
  int m;
  int k;
  int n;
  int lda;
  int ldb;
  int ldc;
  bool transa;
  bool transb;

  int8_t* bias;

  bool operator==(const CublasLtBiasParamsKey& rhs) const {
    return m == rhs.m && k == rhs.k && n == rhs.n && lda == rhs.lda &&
           ldb == rhs.ldb && ldc == rhs.ldc && transa == rhs.transa &&
           transb == rhs.transb && bias == rhs.bias;
  }
};

struct CublasLtBiasParams {
  cublasLtHandle_t ltHandle;
  cublasLtMatmulDesc_t matmulDesc;
  cublasLtMatrixLayout_t Adesc;
  cublasLtMatrixLayout_t Bdesc;
  cublasLtMatrixLayout_t Cdesc;

  CublasLtBiasParams(CublasLtBiasParamsKey& key) {
    cublasLtCreate(&ltHandle);
    cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32I, CUDA_R_32F);

    int m = key.m, k = key.k, n = key.n;

    auto transa_cublas = key.transa ? CUBLAS_OP_T : CUBLAS_OP_N;
    auto transb_cublas = key.transb ? CUBLAS_OP_T : CUBLAS_OP_N;

    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                   &transa_cublas, sizeof(transa_cublas));
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                   &transb_cublas, sizeof(transb_cublas));

    // bias add epilogue
    auto epi = CUBLASLT_EPILOGUE_BIAS;
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                   &epi, sizeof(epi));
    cublasLtMatmulDescSetAttribute(matmulDesc,
                                   CUBLASLT_MATMUL_DESC_BIAS_POINTER, &key.bias,
                                   sizeof(key.bias));
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8I,
                               transa_cublas == CUBLAS_OP_N ? m : k,
                               transa_cublas == CUBLAS_OP_N ? k : m, key.lda);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8I,
                               transb_cublas == CUBLAS_OP_N ? k : n,
                               transb_cublas == CUBLAS_OP_N ? n : k, key.ldb);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_8I, m, n, key.ldc);
  }

  ~CublasLtBiasParams() {
    if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
    if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
    if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
    if (matmulDesc) cublasLtMatmulDescDestroy(matmulDesc);
  }
};

template <class T>
inline void hash_combine(std::size_t& seed, const T& v) {
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

struct CublasLtBiasKeyHasher {
  std::size_t operator()(const CublasLtBiasParamsKey& key) const {
    std::size_t seed = std::hash<int>()(key.m);
    hash_combine(seed, key.k);
    hash_combine(seed, key.n);
    hash_combine(seed, key.lda);
    hash_combine(seed, key.ldb);
    hash_combine(seed, key.ldc);
    hash_combine(seed, key.transa);
    hash_combine(seed, key.transb);
    hash_combine(seed, key.bias);
    return seed;
  }
};

struct CublasLtBiasState : public Context::Resource {
  std::mutex mu;
  std::unordered_map<CublasLtBiasParamsKey, CublasLtBiasParams*,
                     CublasLtBiasKeyHasher>
      cache;
};

void runCublasLtBiasForward(ExecutionContext* ctx, bool transa, bool transb,
                            int m, int n, int k, void const* A, int lda,
                            void const* B, int ldb, void* C, int ldc,
                            void const* alpha, void const* beta,
                            void const* bias) {
  std::string unique_name = "tao_ral.gpu.cublasLt_bias_" +
                            tao::ral::TaoTypeNameHelper<int8_t>::Invoke();
  auto state = ctx->getOrCreateResource<CublasLtBiasState>(
      unique_name, []() { return new CublasLtBiasState; });
  CublasLtBiasParamsKey key{m,   k,      n,      lda,          ldb,
                            ldc, transa, transb, (int8_t*)bias};
  {
    std::lock_guard<std::mutex> l(state->mu);

    auto& cache = state->cache;
    auto it = cache.find(key);
    if (it == cache.end()) {
      auto params_tmp = new CublasLtBiasParams(key);
      cache.insert(std::make_pair(key, params_tmp));
    }
  }
  auto params = state->cache[key];

  auto status =
      cublasLtMatmul(params->ltHandle, params->matmulDesc, alpha, A,
                     params->Adesc, B, params->Bdesc, beta, C, params->Cdesc, C,
                     params->Cdesc, NULL, NULL, 0, 0);
  if (status != CUBLAS_STATUS_SUCCESS) {
    ctx->signalError(Context::FAILURE, "Error to execute cublasLt.");
  }
}

struct GemmScaleKey {
  opaque_t input_scale = nullptr;
  opaque_t weight_scale = nullptr;
  opaque_t output_scale = nullptr;

  bool operator==(const GemmScaleKey& rhs) const {
    return input_scale == rhs.input_scale && weight_scale == rhs.weight_scale &&
           output_scale == rhs.output_scale;
  }
};

struct GemmScaleKeyHasher {
  std::size_t operator()(const GemmScaleKey& key) const {
    std::size_t seed = std::hash<intptr_t>()((intptr_t)key.input_scale);
    hash_combine(seed, key.weight_scale);
    hash_combine(seed, key.output_scale);
    return seed;
  }
};

struct GemmScaleState : public Context::Resource {
  std::mutex mu;
  std::unordered_map<GemmScaleKey, float, GemmScaleKeyHasher> cache;
};

template <int N>
MemRefType<int8_t, N> ral_pdll_qgemm_nt_s8s8s8_biasadd_quant_per_tensor(
    ExecutionContext* ctx, void* stream_handle, MemRefType<int8_t, N> input,
    MemRefType<int8_t, 2> weight, MemRefType<int8_t, 1> bias,
    MemRefType<float, 0> inputScales, MemRefType<int32_t, 0> inputZeroPoints,
    MemRefType<float, 0> weightScales, MemRefType<int32_t, 0> weightZeroPoints,
    MemRefType<float, 0> resultScales, MemRefType<int32_t, 0> resultZeroPoints,
    void* customAttrs) {
  int64_t m = 1;
  int64_t k = weight.sizes[1];
  int64_t n = weight.sizes[0];
  int64_t resultSizes[N];
  for (int i = 0; i < N - 1; i += 1) {
    m *= input.sizes[i];
    resultSizes[i] = input.sizes[i];
  }
  resultSizes[N - 1] = n;

  if (isEmptyMemref(input) || isEmptyMemref(weight)) {
    TAO_VLOG(1) << "ral_qgemm: early return for empty tensor";
    return assignMemRef<int8_t, N>(nullptr, resultSizes);
    ;
  }

  float kernel_scale;
  float beta = 0.0f;

  {
    // Kernel internal scale is calculated by input_scale, weight_scale and
    // result_scale in ral call function. Since these three tensors are stored
    // in device, we have to load them from device to host which will break
    // kernel launch and influence the overall latency. Here we choose to
    // cache the calculated scale to avoid extra h2d overhead.
    GemmScaleKey key{inputScales.data, weightScales.data, resultScales.data};
    std::string unique_name = "tao_ral.gpu.qgemm_scale" +
                              tao::ral::TaoTypeNameHelper<int8_t>::Invoke();
    auto state = ctx->getOrCreateResource<GemmScaleState>(
        unique_name, []() { return new GemmScaleState; });
    std::lock_guard<std::mutex> l(state->mu);
    auto& cache = state->cache;
    auto it = cache.find(key);
    if (it == cache.end()) {
      auto gpu_driver = ctx->getDriver<GPUDriver>(GPUDriver::name());
      float input_scale, weight_scale, result_scale;
      gpu_driver->d2h(ctx, stream_handle, &inputScales.data[0], &input_scale,
                      sizeof(float));
      gpu_driver->d2h(ctx, stream_handle, &weightScales.data[0], &weight_scale,
                      sizeof(float));
      gpu_driver->d2h(ctx, stream_handle, &resultScales.data[0], &result_scale,
                      sizeof(float));

      gpu_driver->syncOnStream(ctx, stream_handle);

      float kernel_scale_tmp = input_scale * weight_scale / result_scale;
      it = cache.insert(std::make_pair(key, kernel_scale_tmp)).first;
    }
    kernel_scale = (float)(it->second);
  }

  // For int8 gemm kernel, matrix a is always row major and matrix b is always
  // column major.
  bool transa = false;
  bool transb = true;

  auto gpu_driver = ctx->getDriver<GPUDriver>(GPUDriver::name());
  auto data =
      static_cast<int8_t*>(gpu_driver->alloc(ctx, m * n * sizeof(int8_t)));
  auto result = assignMemRef<int8_t, N>(data, resultSizes);

#if defined(PLATFORM_ALIBABA) and defined(ENABLE_BLADE_GEMM)
  {
    beta = 1.0f;
    void* s = gpu_driver->asCUStream(ctx, stream_handle);
    bladnn::Context bladnn_ctx{s};
    bladnn::Dtype in_dtype = bladnn::Dtype::kS8;
    ;
    bladnn::Dtype out_dtype = bladnn::Dtype::kS8;
    ;
    bool ret =
        bladnn::gemm(&bladnn_ctx, in_dtype, 0, input.data, m, k, in_dtype, 1,
                     weight.data, n, k, out_dtype, result.data, m, n, 1, false,
                     false, &kernel_scale, &beta, bias.data);
    if (ret) {
      return result;
    }
  }
#endif
  beta = 0.0f;

  runCublasLtBiasForward(ctx, transb, transa, n, m, k, weight.data, k,
                         input.data, k, result.data, n, &kernel_scale, &beta,
                         bias.data);
  return result;
}

}  // namespace se_impl
}  // namespace gpu

TAO_RAL_API("ral_pdll_qgemm", "gpu",
            gpu::se_impl::ral_pdll_qgemm_nt_s8s8s8_biasadd_quant_per_tensor<2>);
TAO_RAL_API("ral_pdll_qgemm", "gpu",
            gpu::se_impl::ral_pdll_qgemm_nt_s8s8s8_biasadd_quant_per_tensor<3>);

}  // namespace ral
}  // namespace tao