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
#include "tensorflow/core/kernels/ant_fused_embedding/dense_kernels.cu.h"

void EmbVecsGather(const GPUDevice& d, const float* emb_table,
                   const int64_t* values, const int batch_size,
                   const float max_norm, const int emb_vec_size,
                   const int fill_empty_row_default_id, float* emb_vectors) {
  const int threads = emb_vec_size;
  const int blocks = batch_size;
  TF_CHECK_OK(GpuLaunchKernel(EmbVecsGatherKernel, blocks, threads, 0,
                              d.stream(), emb_table, values, batch_size,
                              max_norm, emb_vec_size, fill_empty_row_default_id,
                              emb_vectors));
}

}  // namespace ant_fused_embedding

class FusedDenseLocalEmbeddingLookUpGPU : public OpKernel {
 public:
  explicit FusedDenseLocalEmbeddingLookUpGPU(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_norm", &max_norm_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fill_empty_row_default_id",
                                     &fill_empty_row_default_id_));
  }

  void Compute(OpKernelContext* ctx) override {
    using namespace ant_fused_embedding;
    auto device = ctx->eigen_device<GPUDevice>();

    Tensor const* values = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("values", &values));

    Tensor const* emb_table = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("emb_table", &emb_table));

    nvtx::ScopedRangeIfEnabled<nvtx::CoreDomain> nvtx_range(this);

    const int emb_vec_size = emb_table->shape().dim_size(1);
    const int batch_size = values->shape().dim_size(0);

    Tensor* emb_vectors = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("emb_vectors",
                                             TensorShape({int64(batch_size),
                                                          int64(emb_vec_size)}),
                                             &emb_vectors));

    EmbVecsGather(device, data_p_with_type<const float>(emb_table),
                  data_p_with_type<const int64_t>(values), batch_size,
                  max_norm_, emb_vec_size, fill_empty_row_default_id_,
                  data_p_with_type<float>(emb_vectors));
  }

 private:
  float max_norm_;
  int fill_empty_row_default_id_;
};

REGISTER_KERNEL_BUILDER(
    Name("FusedDenseLocalEmbeddingLookUp").Device(DEVICE_GPU),
    FusedDenseLocalEmbeddingLookUpGPU);
}  // namespace tensorflow

#endif  // GOOGLE_CUDA