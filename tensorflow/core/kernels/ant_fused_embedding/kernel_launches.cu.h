#pragma once

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

namespace ant_fused_embedding {

using GPUDevice = Eigen::GpuDevice;

#include "tensorflow/core/kernels/ant_fused_embedding/dense_kernels.cu.h"
#include "tensorflow/core/kernels/ant_fused_embedding/sparse_kernels.cu.h"

inline int CalcBlocksLinearMapping(const int problem_size,
                                   const int thread_cover) {
  return problem_size % thread_cover == 0 ? (problem_size / thread_cover)
                                          : (problem_size / thread_cover + 1);
}

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
  const int threads = 32;
  const int blocks = nnz;
  TF_CHECK_OK(GpuLaunchKernel(
      EmbVecsGatherAndCombineKernel<combiner>, blocks, threads, 0, d.stream(),
      emb_table, sp_values, sp_indices, sub_feature_nums, batch_size, max_norm,
      emb_vec_size, fill_empty_row_default_id, emb_vectors));
}

void EmbVecsGather(const GPUDevice& d, const float* emb_table,
                   const int64_t* values, const int batch_size,
                   const float max_norm, const int emb_vec_size,
                   const int fill_empty_row_default_id, float* emb_vectors) {
  const int threads = 32;
  const int blocks = batch_size;
  TF_CHECK_OK(GpuLaunchKernel(EmbVecsGatherKernel, blocks, threads, 0,
                              d.stream(), emb_table, values, batch_size,
                              max_norm, emb_vec_size, fill_empty_row_default_id,
                              emb_vectors));
}

}  // namespace ant_fused_embedding

}  // namespace tensorflow

#endif  // GOOGLE_CUDA