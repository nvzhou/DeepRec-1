#pragma once

__global__ void InitEmbVecsAndSubFeatureNumVec4Kernel(const int batch_size,
                                                      const int emv_vec_dim,
                                                      float* emb_vectors,
                                                      int* sub_feature_num) {
  // two tasks: init emb_vecs and sub_feature_num
  const int g_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int task_1_bound = batch_size * emv_vec_dim;
  const int task_2_bound = batch_size;
  const int t_num_for_task_1 =
      task_1_bound % 4 == 0 ? (task_1_bound / 4) : (task_1_bound / 4 + 1);
  if (4 * g_tid < task_1_bound) {
    // belongs to task 1
    if (4 * g_tid + 3 < task_1_bound) {
      *((float4*)(emb_vectors + 4 * g_tid)) =
          make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    } else {
      for (int i = 0; i < task_1_bound - 4 * g_tid; i++) {
        emb_vectors[4 * g_tid + i] = 0.0f;
      }
    }
  } else {
    // belong to task 2
    const int g_tid_task_2 = g_tid - t_num_for_task_1;
    if (4 * g_tid_task_2 + 3 < task_2_bound) {
      *((int4*)(sub_feature_num + 4 * g_tid_task_2)) = make_int4(0, 0, 0, 0);
    } else {
      for (int i = 0; i < task_2_bound - 4 * g_tid_task_2; i++) {
        sub_feature_num[4 * g_tid_task_2 + i] = 0;
      }
    }
  }
}

__global__ void GetSubFeatureNumKernel(const int64_t* sp_values,
                                       const int64_t* sp_indices, const int nnz,
                                       int* sub_feature_nums) {
  /*
  1. Naturally prune invalid ids. Otherwise EmbVecsGatherAndCombineKernel may
  crash.
  */
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < nnz) {
    const int64_t key = sp_values[gid];
    const int64_t row_in_batch = sp_indices[2 * gid];
    if (key >= 0) {  // prune invalid ids
      atomicAdd(sub_feature_nums + row_in_batch, 1);
    }
  }
}

enum Combiner { Mean, Sum, Sqrtn };

template <Combiner combiner>
__forceinline__ __device__ float Combine(const float in, const int feature_num);

template <>
__forceinline__ __device__ float Combine<Sqrtn>(const float in,
                                                const int sub_feature_num) {
  return in / sqrtf(sub_feature_num);
}

template <>
__forceinline__ __device__ float Combine<Mean>(const float in,
                                               const int sub_feature_num) {
  return in / sub_feature_num;
}

template <>
__forceinline__ __device__ float Combine<Sum>(const float in,
                                              const int sub_feature_num) {
  return in;
}

template <Combiner combiner>
__forceinline__ __device__ void InternalMaxNormCombineAndWriteOutput(
    float emb_element, const int64_t row_in_batch, const int emb_vec_size,
    const float max_norm, const int sub_feature_num, float* l2_sum,
    float* emb_vectors) {
  if (max_norm >= 0.0f) {
    if (threadIdx.x == 0) {
      l2_sum[0] = 0.0f;
    }
    __syncthreads();
    atomicAdd(l2_sum, emb_element * emb_element);
    __syncthreads();
    float l2_norm = sqrtf(l2_sum[0]);
    if (l2_norm > max_norm) {
      emb_element *= max_norm / l2_norm;
    }
  }

  emb_element = Combine<combiner>(emb_element, sub_feature_num);
  atomicAdd(emb_vectors + row_in_batch * emb_vec_size + threadIdx.x,
            emb_element);
}

template <Combiner combiner>
__global__ void EmbVecsGatherAndCombineKernel(
    const float* emb_table, const int64_t* sp_values, const int64_t* sp_indices,
    const int* sub_feature_nums, const int batch_size, const float max_norm,
    const int emb_vec_size, const int fill_empty_row_default_id,
    float* emb_vectors) {
  /*
  1. Naturally prune invalid ids, otherwise if allows id < 0, the kernel may
  just crash.
  2. TF 1.x equivalent.
      if fill_empty_row_default_id < 0:
         will just fill the empty row in emb vectors with all 0.0f, which is
         aleady done in InitEmbVecsAndSubFeatureNumVec4Kernel.
      else: fill the empty row in emb vectors with the values of
         emb_table[fill_empty_row_default_id].
  */

  __shared__ float l2_sum[1];

  const int64_t key = sp_values[blockIdx.x];

  if (key >= 0) {
    const int64_t row_in_batch = sp_indices[2 * blockIdx.x];
    const int sub_feature_num = sub_feature_nums[row_in_batch];
    float emb_element = emb_table[key * emb_vec_size + threadIdx.x];
    InternalMaxNormCombineAndWriteOutput<combiner>(
        emb_element, row_in_batch, emb_vec_size, max_norm, sub_feature_num,
        l2_sum, emb_vectors);
  }

  if (fill_empty_row_default_id >= 0) {
    for (int i = blockIdx.x; i < batch_size; i += blockDim.x) {
      if (sub_feature_nums[i] == 0) {
        // it's an empty row
        float emb_element =
            emb_table[fill_empty_row_default_id * emb_vec_size + threadIdx.x];
        InternalMaxNormCombineAndWriteOutput<combiner>(
            emb_element, i, emb_vec_size, max_norm, 1, l2_sum, emb_vectors);
      }
    }
  }
}