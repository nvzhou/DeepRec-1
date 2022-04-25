#include <exception>
#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/ant_fused_embedding/common.cu.h"
#include "tensorflow/core/profiler/nvtx_utils.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "third_party/cub/thread/thread_operators.cuh"

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;

namespace ant_fused_embedding {
#include "tensorflow/core/kernels/ant_fused_embedding/sparse_kernels.cu.h"

void InitEmbVecsAndSubFeatureNumVec4(const GPUDevice& d, const int batch_size,
                                     const int emv_vec_dim, float* emb_vectors,
                                     int* sub_feature_num) {
  const int threads = 32;
  const int blocks =
      CalcBlocksLinearMapping(batch_size * emv_vec_dim, threads * 4) +
      CalcBlocksLinearMapping(batch_size, threads * 4);

  TF_CHECK_OK(GpuLaunchKernel(InitEmbVecsAndSubFeatureNumVec4Kernel, blocks,
                              threads, 0, d.stream(), batch_size, emv_vec_dim,
                              emb_vectors, sub_feature_num));
}

void GetSubFeatureNum(const GPUDevice& d, const int64_t* sp_values,
                      const int64_t* sp_indices, const int nnz,
                      int* sub_feature_nums) {
  const int threads = 32;
  const int blocks = CalcBlocksLinearMapping(nnz, threads);
  TF_CHECK_OK(GpuLaunchKernel(GetSubFeatureNumKernel, blocks, threads, 0,
                              d.stream(), sp_values, sp_indices, nnz,
                              sub_feature_nums));
}

template <Combiner combiner>
void EmbVecsGatherAndCombine(
    const GPUDevice& d, const float* emb_table, const int64_t* sp_values,
    const int64_t* sp_indices, const int* sub_feature_nums, const int nnz,
    const int batch_size, const float max_norm, const int emb_vec_size,
    const int fill_empty_row_default_id, float* emb_vectors) {
  const int threads = emb_vec_size;
  const int blocks = nnz;
  TF_CHECK_OK(GpuLaunchKernel(
      EmbVecsGatherAndCombineKernel<combiner>, blocks, threads, 0, d.stream(),
      emb_table, sp_values, sp_indices, sub_feature_nums, batch_size, max_norm,
      emb_vec_size, fill_empty_row_default_id, emb_vectors));
}

}  // namespace ant_fused_embedding

class FusedSparseLocalEmbeddingLookUpGPU : public OpKernel {
 public:
  explicit FusedSparseLocalEmbeddingLookUpGPU(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("combiner", &combiner_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_norm", &max_norm_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fill_empty_row_default_id",
                                     &fill_empty_row_default_id_));
  }

  void Compute(OpKernelContext* ctx) override {
    using namespace ant_fused_embedding;
    auto device = ctx->eigen_device<GPUDevice>();

    Tensor const* sp_values = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sp_values", &sp_values));

    Tensor const* sp_indices = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sp_indices", &sp_indices));

    Tensor const* sp_dense_shape = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sp_dense_shape", &sp_dense_shape));

    Tensor const* emb_table = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("emb_table", &emb_table));

    nvtx::ScopedRangeIfEnabled<nvtx::CoreDomain> nvtx_range(this);

    const int emb_vec_size = emb_table->shape().dim_size(1);
    const int batch_size = sp_dense_shape->flat<int64>().data()[0];
    const int nnz = sp_values->shape().dim_size(0);

    Tensor sub_feature_nums;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(DT_INT32, TensorShape({int64(batch_size)}),
                                &sub_feature_nums));

    Tensor* emb_vectors = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("emb_vectors",
                                             TensorShape({int64(batch_size),
                                                          int64(emb_vec_size)}),
                                             &emb_vectors));

    InitEmbVecsAndSubFeatureNumVec4(device, batch_size, emb_vec_size,
                                    data_p_with_type<float>(emb_vectors),
                                    data_p_with_type<int>(sub_feature_nums));

    GetSubFeatureNum(device, data_p_with_type<const int64_t>(sp_values),
                     data_p_with_type<const int64_t>(sp_indices), nnz,
                     data_p_with_type<int>(sub_feature_nums));

    if (combiner_ == "sqrtn") {
      EmbVecsGatherAndCombine<Sqrtn>(
          device, data_p_with_type<const float>(emb_table),
          data_p_with_type<const int64_t>(sp_values),
          data_p_with_type<const int64_t>(sp_indices),
          data_p_with_type<const int>(sub_feature_nums), nnz, batch_size,
          max_norm_, emb_vec_size, fill_empty_row_default_id_,
          data_p_with_type<float>(emb_vectors));
    } else if (combiner_ == "mean") {
      EmbVecsGatherAndCombine<Mean>(
          device, data_p_with_type<const float>(emb_table),
          data_p_with_type<const int64_t>(sp_values),
          data_p_with_type<const int64_t>(sp_indices),
          data_p_with_type<const int>(sub_feature_nums), nnz, batch_size,
          max_norm_, emb_vec_size, fill_empty_row_default_id_,
          data_p_with_type<float>(emb_vectors));
    } else {
      EmbVecsGatherAndCombine<Sum>(
          device, data_p_with_type<const float>(emb_table),
          data_p_with_type<const int64_t>(sp_values),
          data_p_with_type<const int64_t>(sp_indices),
          data_p_with_type<const int>(sub_feature_nums), nnz, batch_size,
          max_norm_, emb_vec_size, fill_empty_row_default_id_,
          data_p_with_type<float>(emb_vectors));
    }
  }

 private:
  std::string combiner_;
  float max_norm_;
  int fill_empty_row_default_id_;
};

REGISTER_KERNEL_BUILDER(Name("FusedSparseLocalEmbeddingLookUp")
                            .Device(DEVICE_GPU)
                            .HostMemory("sp_dense_shape"),
                        FusedSparseLocalEmbeddingLookUpGPU);
}  // namespace tensorflow

#endif  // GOOGLE_CUDA