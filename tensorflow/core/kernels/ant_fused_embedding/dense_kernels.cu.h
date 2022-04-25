#pragma once

__forceinline__ __device__ void InternalMaxNormAndWriteOutput(
    float emb_element, const int emb_vec_size, const float max_norm,
    float* l2_sum, float* emb_vectors) {
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

  emb_vectors[blockIdx.x * emb_vec_size + threadIdx.x] = emb_element;
}

__global__ void EmbVecsGatherKernel(const float* emb_table,
                                    const int64_t* values, const int batch_size,
                                    const float max_norm,
                                    const int emb_vec_size,
                                    const int fill_empty_row_default_id,
                                    float* emb_vectors) {
  /*
   */
  __shared__ float l2_sum[1];
  const int64_t key = values[blockIdx.x];
  if (key > 0) {
    float emb_element = emb_table[key * emb_vec_size + threadIdx.x];
    InternalMaxNormAndWriteOutput(emb_element, emb_vec_size, max_norm, l2_sum,
                                  emb_vectors);
  } else {
    // invalid key
    if (fill_empty_row_default_id > 0) {
      // set it to corresponding values of fill_empty_row_default_id
      float emb_element =
          emb_table[fill_empty_row_default_id * emb_vec_size + threadIdx.x];
      InternalMaxNormAndWriteOutput(emb_element, emb_vec_size, max_norm, l2_sum,
                                    emb_vectors);
    } else {
      // set to 0.0f
      emb_vectors[blockIdx.x * emb_vec_size + threadIdx.x] = 0.0f;
    }
  }
}