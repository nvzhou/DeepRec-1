#include <stdio.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("FusedSparseLocalEmbeddingLookUp")
    .Attr("T : {float32}")
    .Attr("fill_empty_row_default_id: int = -1")
    .Attr("combiner: {'sqrtn', 'mean', 'sum'}")
    .Attr("max_norm: float = -1.0")
    .Input("sp_values: int64")
    .Input("sp_indices: int64")
    .Input("sp_dense_shape: int64")
    .Input("emb_table: T")
    .Output("emb_vectors: T")
    .SetShapeFn([](InferenceContext* ctx) {
      std::vector<ShapeHandle> unused_list;
      ShapeHandle unused;
      DimensionHandle unused_dim;

      // sp_values
      ctx->input("sp_values", &unused_list);
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 1, &unused));

      // sp_indices
      ctx->input("sp_indices", &unused_list);
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 2, &unused));

      // sp_dense_shape
      ctx->input("sp_dense_shape", &unused_list);
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 1, &unused));

      // emb_table
      ctx->input("emb_table", &unused_list);
      ShapeHandle emb_table_shape;
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 2, &emb_table_shape));
      DimensionHandle emb_vec_size_dim = ctx->Dim(emb_table_shape, 1);
      int64 emb_vec_size = ctx->Value(emb_vec_size_dim);

      // emb_vectors
      unused_list.clear();
      unused_list.resize(1);
      unused_list[0] = ctx->MakeShape({ctx->UnknownDim(), emb_vec_size_dim});
      ctx->set_output("emb_vectors", unused_list);

      return Status::OK();
    });

REGISTER_OP("FusedDenseLocalEmbeddingLookUp")
    .Attr("T : {float32}")
    .Attr("fill_empty_row_default_id: int = -1")
    .Attr("max_norm: float = -1.0")
    .Input("values: int64")
    .Input("emb_table: T")
    .Output("emb_vectors: T")
    .SetShapeFn([](InferenceContext* ctx) {
      std::vector<ShapeHandle> unused_list;
      ShapeHandle unused;
      DimensionHandle unused_dim;

      // sp_values
      ctx->input("values", &unused_list);
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 1, &unused));

      // emb_table
      ctx->input("emb_table", &unused_list);
      ShapeHandle emb_table_shape;
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 2, &emb_table_shape));
      DimensionHandle emb_vec_size_dim = ctx->Dim(emb_table_shape, 1);
      int64 emb_vec_size = ctx->Value(emb_vec_size_dim);

      // emb_vectors
      unused_list.clear();
      unused_list.resize(1);
      unused_list[0] = ctx->MakeShape({ctx->UnknownDim(), emb_vec_size_dim});
      ctx->set_output("emb_vectors", unused_list);

      return Status::OK();
    });


}  // namespace tensorflow