#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace {

enum class Device { GPU };
class FusedSparseLocalEmbeddingLookUpTest : public OpsTestBase {
 protected:
  void MakeOpAndSetDevice(Device device, DataType dtype,
                          const std::string& combiner, const float max_norm,
                          const int fill_empty_row_default_id) {
    if (device == Device::GPU) {
      SetDevice(DEVICE_GPU,
                std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                    "GPU", {}, "/job:a/replica:0/task:0")));
    }

    TF_EXPECT_OK(
        NodeDefBuilder("fused_sparse_local_embedding_look_up",
                       "FusedSparseLocalEmbeddingLookUp")
            .Attr("T", dtype)
            .Attr("combiner", combiner)
            .Attr("max_norm", max_norm)
            .Attr("fill_empty_row_default_id", fill_empty_row_default_id)
            .Input(FakeInput(DT_INT64))
            .Input(FakeInput(DT_INT64))
            .Input(FakeInput(DT_INT64))
            .Input(FakeInput(dtype))
            .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

TEST_F(FusedSparseLocalEmbeddingLookUpTest, SqrtnMaxNorm200) {
  const int nnz = 10;
  const int batch_size = 4;
  const int emb_vector_dim = 8;
  const int bucket_size = 16;
  const int entries = 8;

  MakeOpAndSetDevice(Device::GPU, DT_FLOAT, "sqrtn", 200.0, -1);

  // sp_values
  AddInputFromArray<int64>(TensorShape({nnz}),
                           {3, 1, 4, 5, 7, 3, 12, 12, 15, 4});
  // sp_indices
  AddInputFromArray<int64>(
      TensorShape({nnz, 2}),
      {0, 1, 0, 5, 1, 2, 1, 1, 1, 7, 2, 1, 2, 4, 2, 7, 3, 0, 3, 6});
  // sp_dense_shape
  AddInputFromArray<int64>(TensorShape({2}), {batch_size, entries});

  // emb_table
  AddInputFromArray<float>(
      TensorShape({bucket_size, emb_vector_dim}),
      {0.0,   1.0,   2.0,   3.0,   4.0,   5.0,   6.0,   7.0,   8.0,   9.0,
       10.0,  11.0,  12.0,  13.0,  14.0,  15.0,  16.0,  17.0,  18.0,  19.0,
       20.0,  21.0,  22.0,  23.0,  24.0,  25.0,  26.0,  27.0,  28.0,  29.0,
       30.0,  31.0,  32.0,  33.0,  34.0,  35.0,  36.0,  37.0,  38.0,  39.0,
       40.0,  41.0,  42.0,  43.0,  44.0,  45.0,  46.0,  47.0,  48.0,  49.0,
       50.0,  51.0,  52.0,  53.0,  54.0,  55.0,  56.0,  57.0,  58.0,  59.0,
       60.0,  61.0,  62.0,  63.0,  64.0,  65.0,  66.0,  67.0,  68.0,  69.0,
       70.0,  71.0,  72.0,  73.0,  74.0,  75.0,  76.0,  77.0,  78.0,  79.0,
       80.0,  81.0,  82.0,  83.0,  84.0,  85.0,  86.0,  87.0,  88.0,  89.0,
       90.0,  91.0,  92.0,  93.0,  94.0,  95.0,  96.0,  97.0,  98.0,  99.0,
       100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0,
       110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0,
       120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  {
    Tensor expected_emb_vectors(allocator(), DT_FLOAT,
                                TensorShape({batch_size, emb_vector_dim}));
    test::FillValues<float>(
        &expected_emb_vectors,
        {22.62741661, 24.04163170, 25.45584488,  26.87005806,  28.28427124,
         29.69848442, 31.11269951, 32.52691269,  73.90083313,  75.63288879,
         77.36493683, 79.09698486, 80.82904053,  82.56108856,  84.29314423,
         86.02519226, 92.61308289, 94.01081848,  95.40855408,  96.80628204,
         98.20401764, 99.60175323, 100.99948120, 102.39721680, 71.20205688,
         72.31395721, 73.42584991, 74.53774261,  75.64963531,  76.76153564,
         77.87342834, 78.98532867});
    test::ExpectTensorNear<float>(expected_emb_vectors, *GetOutput(0), 1e-4);
  }
}

TEST_F(FusedSparseLocalEmbeddingLookUpTest, InvalidMeanNoDefaultId) {
  const int nnz = 10;
  const int batch_size = 4;
  const int emb_vector_dim = 4;
  const int bucket_size = 16;
  const int entries = 8;

  MakeOpAndSetDevice(Device::GPU, DT_FLOAT, "mean", -1.0f, -1);

  // sp_values
  AddInputFromArray<int64>(TensorShape({nnz}),
                           {-3, -1, -4, 5, 7, 3, 12, 12, 15, 4});
  // sp_indices
  AddInputFromArray<int64>(
      TensorShape({nnz, 2}),
      {0, 1, 0, 5, 1, 2, 1, 1, 1, 7, 2, 1, 2, 4, 2, 7, 3, 0, 3, 6});
  // sp_dense_shape
  AddInputFromArray<int64>(TensorShape({2}), {batch_size, entries});

  // emb_table
  AddInputFromArray<float>(
      TensorShape({bucket_size, emb_vector_dim}),
      {
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
      });

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  {
    Tensor expected_emb_vectors(allocator(), DT_FLOAT,
                                TensorShape({batch_size, emb_vector_dim}));
    test::FillValues<float>(&expected_emb_vectors,
                            {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
    test::ExpectTensorNear<float>(expected_emb_vectors, *GetOutput(0), 1e-4);
  }
}

TEST_F(FusedSparseLocalEmbeddingLookUpTest, InvalidSumDefaultId) {
  const int nnz = 10;
  const int batch_size = 4;
  const int emb_vector_dim = 4;
  const int bucket_size = 16;
  const int entries = 8;

  MakeOpAndSetDevice(Device::GPU, DT_FLOAT, "sum", -1.0f, 2);

  // sp_values
  AddInputFromArray<int64>(TensorShape({nnz}),
                           {-3, -1, -4, 5, 7, 3, 12, 12, 15, 4});
  // sp_indices
  AddInputFromArray<int64>(
      TensorShape({nnz, 2}),
      {0, 1, 0, 5, 1, 2, 1, 1, 1, 7, 2, 1, 2, 4, 2, 7, 3, 0, 3, 6});
  // sp_dense_shape
  AddInputFromArray<int64>(TensorShape({2}), {batch_size, entries});

  // emb_table
  AddInputFromArray<float>(
      TensorShape({bucket_size, emb_vector_dim}),
      {
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 99.0, 99.0, 99.0, 99.0, 1.0,
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  1.0,  1.0,  1.0,  1.0,
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  1.0,  1.0,  1.0,  1.0,
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  1.0,  1.0,  1.0,  1.0,
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  1.0,  1.0,  1.0,
      });

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  {
    Tensor expected_emb_vectors(allocator(), DT_FLOAT,
                                TensorShape({batch_size, emb_vector_dim}));
    test::FillValues<float>(&expected_emb_vectors,
                            {99.0, 99.0, 99.0, 99.0, 2.0, 2.0, 2.0,
                             2.0, 3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0, 2.0});
    test::ExpectTensorNear<float>(expected_emb_vectors, *GetOutput(0), 1e-4);
  }
}

}  // namespace
}  // namespace tensorflow