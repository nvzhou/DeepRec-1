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
class FusedDenseLocalEmbeddingLookUpTest : public OpsTestBase {
 protected:
  void MakeOpAndSetDevice(Device device, DataType dtype, const float max_norm,
                          const int fill_empty_row_default_id) {
    if (device == Device::GPU) {
      SetDevice(DEVICE_GPU,
                std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                    "GPU", {}, "/job:a/replica:0/task:0")));
    }

    TF_EXPECT_OK(
        NodeDefBuilder("fused_dense_local_embedding_look_up",
                       "FusedDenseLocalEmbeddingLookUp")
            .Attr("T", dtype)
            .Attr("max_norm", max_norm)
            .Attr("fill_empty_row_default_id", fill_empty_row_default_id)
            .Input(FakeInput(DT_INT64))
            .Input(FakeInput(dtype))
            .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

TEST_F(FusedDenseLocalEmbeddingLookUpTest, MaxNorm10) {
  const int batch_size = 4;
  const int emb_vector_dim = 4;
  const int bucket_size = 8;

  MakeOpAndSetDevice(Device::GPU, DT_FLOAT, 10.0, -1);

  // values
  AddInputFromArray<int64>(TensorShape({batch_size}), {2, 4, 1, 6});

  // emb_table
  AddInputFromArray<float>(
      TensorShape({bucket_size, emb_vector_dim}),
      {1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0,
       3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0,
       6.0, 6.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  {
    Tensor expected_emb_vectors(allocator(), DT_FLOAT,
                                TensorShape({batch_size, emb_vector_dim}));
    test::FillValues<float>(&expected_emb_vectors,
                            {3.0, 3.0, 3.0, 3.0, 5.0, 5.0, 5.0, 5.0, 2.0, 2.0,
                             2.0, 2.0, 5.0, 5.0, 5.0, 5.0});
    test::ExpectTensorNear<float>(expected_emb_vectors, *GetOutput(0), 1e-4);
  }
}

TEST_F(FusedDenseLocalEmbeddingLookUpTest, InvalidNoDefaultId) {
  const int batch_size = 4;
  const int emb_vector_dim = 4;
  const int bucket_size = 8;

  MakeOpAndSetDevice(Device::GPU, DT_FLOAT, -1.0f, -1);

  // values
  AddInputFromArray<int64>(TensorShape({batch_size}), {2, -4, 1, 6});

  // emb_table
  AddInputFromArray<float>(
      TensorShape({bucket_size, emb_vector_dim}),
      {1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0,
       3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0,
       6.0, 6.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  {
    Tensor expected_emb_vectors(allocator(), DT_FLOAT,
                                TensorShape({batch_size, emb_vector_dim}));
    test::FillValues<float>(&expected_emb_vectors,
                            {3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0,
                             2.0, 2.0, 7.0, 7.0, 7.0, 7.0});
    test::ExpectTensorNear<float>(expected_emb_vectors, *GetOutput(0), 1e-4);
  }
}

TEST_F(FusedDenseLocalEmbeddingLookUpTest, InvalidDefaultId) {
  const int batch_size = 4;
  const int emb_vector_dim = 4;
  const int bucket_size = 8;

  MakeOpAndSetDevice(Device::GPU, DT_FLOAT, -1.0f, 3);

  // values
  AddInputFromArray<int64>(TensorShape({batch_size}), {2, -4, 1, 6});

  // emb_table
  AddInputFromArray<float>(
      TensorShape({bucket_size, emb_vector_dim}),
      {1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0,
       3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0,
       6.0, 6.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  {
    Tensor expected_emb_vectors(allocator(), DT_FLOAT,
                                TensorShape({batch_size, emb_vector_dim}));
    test::FillValues<float>(&expected_emb_vectors,
                            {3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 2.0, 2.0,
                             2.0, 2.0, 7.0, 7.0, 7.0, 7.0});
    test::ExpectTensorNear<float>(expected_emb_vectors, *GetOutput(0), 1e-4);
  }
}

}  // namespace
}  // namespace tensorflow