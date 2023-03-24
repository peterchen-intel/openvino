// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"

#include "common_op_table.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/opsets/opset9.hpp"

using namespace std;
using namespace ov;
using namespace ov::frontend::tensorflow;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

#define TF_OP_CONVERTER(op) OutputVector op(const ov::frontend::tensorflow::NodeContext& node)

TF_OP_CONVERTER(translate_if_op);
TF_OP_CONVERTER(translate_block_lstm_op);
TF_OP_CONVERTER(translate_fifo_queue_op);
TF_OP_CONVERTER(translate_gru_block_cell_op);
TF_OP_CONVERTER(translate_hash_table_op);
TF_OP_CONVERTER(translate_iterator_get_next_op);
TF_OP_CONVERTER(translate_iterator_op);
TF_OP_CONVERTER(translate_partitioned_call_op);
TF_OP_CONVERTER(translate_queue_dequeue_op);
TF_OP_CONVERTER(translate_queue_dequeue_many_op);
TF_OP_CONVERTER(translate_sparse_fill_empty_rows_op);
TF_OP_CONVERTER(translate_sparse_reshape_op);
TF_OP_CONVERTER(translate_sparse_segment_sum_op);
TF_OP_CONVERTER(translate_while_op);

const std::map<std::string, CreatorFunction> get_supported_ops() {
    return {
        // note: UnaryOp translator declaration for each op must to be added in unary_op.cpp file
        {"Abs", translate_unary_op<opset8::Abs>},
        {"Acos", translate_unary_op<opset8::Acos>},
        {"Acosh", translate_unary_op<opset8::Acosh>},
        {"Asin", translate_unary_op<opset8::Asin>},
        {"Asinh", translate_unary_op<opset8::Asinh>},
        {"Atan", translate_unary_op<opset8::Atan>},
        {"Atanh", translate_unary_op<opset8::Atanh>},
        {"Ceil", translate_unary_op<opset8::Ceiling>},
        {"Cos", translate_unary_op<opset8::Cos>},
        {"Cosh", translate_unary_op<opset8::Cosh>},
        {"Erf", translate_unary_op<opset8::Erf>},
        {"Exp", translate_unary_op<opset8::Exp>},
        {"Floor", translate_unary_op<opset8::Floor>},
        {"IsFinite", translate_unary_op<opset10::IsFinite>},
        {"IsInf", translate_unary_op<opset10::IsInf>},
        {"IsNan", translate_unary_op<opset10::IsNaN>},
        {"Log", translate_unary_op<opset8::Log>},
        {"LogicalNot", translate_unary_op<opset8::LogicalNot>},
        {"Mish", translate_unary_op<opset8::Mish>},
        {"Neg", translate_unary_op<opset8::Negative>},
        {"Relu", translate_unary_op<opset8::Relu>},
        {"Sigmoid", translate_unary_op<opset8::Sigmoid>},
        {"Sin", translate_unary_op<opset8::Sin>},
        {"Sinh", translate_unary_op<opset8::Sinh>},
        {"Sign", translate_unary_op<opset8::Sign>},
        {"Softplus", translate_unary_op<opset8::SoftPlus>},
        {"Softsign", translate_unary_op<opset9::SoftSign>},
        {"Tan", translate_unary_op<opset8::Tan>},
        {"Tanh", translate_unary_op<opset8::Tanh>},
        {"Swish", translate_unary_op<opset8::Swish>},

        // note: BinaryOp translator declaration for each op must to be added in binary_op.cpp file
        {"Add", translate_binary_op<opset8::Add>},
        {"AddV2", translate_binary_op<opset8::Add>},
        {"Equal", translate_binary_op<opset8::Equal>},
        {"FloorMod", translate_binary_op<opset8::FloorMod>},
        {"Greater", translate_binary_op<opset8::Greater>},
        {"GreaterEqual", translate_binary_op<opset8::GreaterEqual>},
        {"Less", translate_binary_op<opset8::Less>},
        {"LessEqual", translate_binary_op<opset8::LessEqual>},
        {"LogicalAnd", translate_binary_op<opset8::LogicalAnd>},
        {"LogicalOr", translate_binary_op<opset8::LogicalOr>},
        {"LogicalXor", translate_binary_op<opset8::LogicalXor>},
        {"Maximum", translate_binary_op<opset8::Maximum>},
        {"Minimum", translate_binary_op<opset8::Minimum>},
        {"Mul", translate_binary_op<opset8::Multiply>},
        {"Mod", translate_binary_op<opset8::Mod>},
        {"NotEqual", translate_binary_op<opset8::NotEqual>},
        {"Pow", translate_binary_op<opset8::Power>},
        {"RealDiv", translate_binary_op<opset8::Divide>},
        {"SquaredDifference", translate_binary_op<opset8::SquaredDifference>},
        {"Sub", translate_binary_op<opset8::Subtract>},

        // note: ReduceOp translator declaration for each op must to be added in reduce.cpp file
        {"Any", translate_direct_reduce_op<opset8::ReduceLogicalOr>},
        {"All", translate_direct_reduce_op<opset8::ReduceLogicalAnd>},
        {"EuclideanNorm", translate_direct_reduce_op<opset8::ReduceL2>},
        {"Max", translate_direct_reduce_op<opset8::ReduceMax>},
        {"Mean", translate_direct_reduce_op<opset8::ReduceMean>},
        {"Min", translate_direct_reduce_op<opset8::ReduceMin>},
        {"Prod", translate_direct_reduce_op<opset8::ReduceProd>},
        {"Sum", translate_direct_reduce_op<opset8::ReduceSum>},

        // Separate translators:
        {"AddN", translate_add_n_op},
        {"ArgMax", translate_arg_max_op},
        {"ArgMin", translate_arg_min_op},
        {"Assert", translate_no_op},
        {"AvgPool", translate_avg_pool_op},
        {"AvgPool3D", translate_avg_pool_op},
        {"BatchMatMul", translate_batch_mat_mul_op},
        {"BatchMatMulV2", translate_batch_mat_mul_op},
        {"BatchToSpaceND", translate_batch_to_space_nd_op},
        {"BroadcastArgs", translate_broadcast_args_op},
        {"BroadcastTo", translate_broadcast_to_op},
        {"Bucketize", translate_bucketize_op},
        {"BiasAdd", translate_bias_add_op},
        {"Cast", translate_cast_op},
        {"ClipByValue", translate_clip_by_value_op},
        {"Concat", translate_concat_op},
        {"ConcatV2", translate_concat_op},
        {"Const", translate_const_op},
        {"Conv2D", translate_conv_2d_op},
        {"Conv2DBackpropInput", translate_conv_2d_backprop_input_op},
        {"Conv3D", translate_conv_3d_op},
        {"Conv3DBackpropInputV2", translate_conv_3d_backprop_input_v2_op},
        {"CropAndResize", translate_crop_and_resize_op},
        {"CTCGreedyDecoder", translate_ctc_greedy_decoder_op},
        {"CTCLoss", translate_ctc_loss_op},
        {"Cumsum", translate_cumsum_op},
        {"DepthToSpace", translate_depth_to_space_op},
        {"DepthwiseConv2dNative", translate_depthwise_conv_2d_native_op},
        {"DynamicPartition", translate_dynamic_partition_op},
        {"Einsum", translate_einsum_op},
        {"Elu", translate_elu_op},
        {"EmptyTensorList", translate_tensor_list_reserve_op},
        {"ExpandDims", translate_expand_dims_op},
        {"ExtractImagePatches", translate_extract_image_patches_op},
        {"FakeQuantWithMinMaxVars", translate_fake_quant_op},
        {"FakeQuantWithMinMaxVarsPerChannel", translate_fake_quant_op},
        {"FIFOQueue", translate_fifo_queue_op},
        {"FIFOQueueV2", translate_fifo_queue_op},
        {"Fill", translate_fill_op},
        {"FloorDiv", translate_floor_div_op},
        {"FusedBatchNorm", translate_fused_batch_norm_op},
        {"FusedBatchNormV2", translate_fused_batch_norm_op},
        {"FusedBatchNormV3", translate_fused_batch_norm_op},
        {"Gather", translate_gather_op},
        {"GatherV2", translate_gather_v2_op},
        {"GatherNd", translate_gather_nd_op},
        {"HashTable", translate_hash_table_op},
        {"HashTableV2", translate_hash_table_op},
        {"Identity", translate_identity_op},
        {"IdentityN", translate_identity_n_op},
        {"If", translate_if_op},
        {"input_arg", translate_input_arg_op},
        {"Iterator", translate_iterator_op},
        {"IteratorGetNext", translate_iterator_get_next_op},
        {"IteratorV2", translate_iterator_op},
        {"output_arg", translate_output_arg_op},
        {"L2Loss", translate_l2_loss_op},
        {"LeakyRelu", translate_leaky_relu_op},
        {"LinSpace", translate_linspace_op},
        {"ListDiff", translate_list_diff_op},
        {"LogSoftmax", translate_log_softmax_op},
        {"Log1p", translate_log_1p_op},
        {"LookupTableInsert", translate_no_op},
        {"LookupTableInsertV2", translate_no_op},
        {"LRN", translate_lrn_op},
        {"MatMul", translate_mat_mul_op},
        {"MatrixDiag", translate_matrix_diag_op},
        {"MaxPool", translate_max_pool_op},
        {"MaxPoolV2", translate_max_pool_op},
        {"MaxPool3D", translate_max_pool_op},
        {"MirrorPad", translate_mirror_pad_op},
        {"MutableHashTable", translate_hash_table_op},
        {"MutableHashTableV2", translate_hash_table_op},
        {"NonMaxSuppression", translate_non_max_suppression_op},
        {"NonMaxSuppressionV2", translate_non_max_suppression_op},
        {"NonMaxSuppressionV3", translate_non_max_suppression_op},
        {"NonMaxSuppressionV4", translate_non_max_suppression_op},
        {"NonMaxSuppressionV5", translate_non_max_suppression_op},
        {"NoOp", translate_no_op},  // do nothing
        {"OneHot", translate_one_hot_op},
        {"OneShotIterator", translate_iterator_op},
        {"Pack", translate_pack_op},
        {"Pad", translate_pad_op},
        {"PadV2", translate_padv2_op},
        {"QueueDequeue", translate_queue_dequeue_op},
        {"QueueDequeueV2", translate_queue_dequeue_op},
        {"QueueDequeueUpTo", translate_queue_dequeue_many_op},
        {"QueueDequeueUpToV2", translate_queue_dequeue_many_op},
        {"QueueDequeueMany", translate_queue_dequeue_many_op},
        {"DynamicStitch", translate_parallel_dynamic_stitch_op},
        {"ParallelDynamicStitch", translate_parallel_dynamic_stitch_op},
        {"PartitionedCall", translate_partitioned_call_op},
        {"Placeholder", translate_placeholder_op},
        {"PlaceholderWithDefault", translate_placeholder_with_default_op},
        {"PreventGradient", translate_identity_op},
        {"Range", translate_range_op},
        {"Rank", translate_rank_op},
        {"RandomUniform", translate_random_uniform_op},
        {"RandomUniformInt", translate_random_uniform_int_op},
        {"Reciprocal", translate_reciprocal_op},
        {"Relu6", translate_relu_6_op},
        {"Reshape", translate_reshape_op},
        {"Reverse", translate_reverse_op},
        {"ReverseSequence", translate_reverse_sequence_op},
        {"ReverseV2", translate_reverse_v2_op},
        {"ResizeBilinear", translate_interpolate_op},
        {"ResizeNearestNeighbor", translate_interpolate_op},
        {"ResourceGather", translate_resource_gather_op},
        {"Roll", translate_roll_op},
        {"Round", translate_round_op},
        {"Rsqrt", translate_rsqrt_op},
        {"SaveV2", translate_no_op},
        {"ScatterNd", translate_scatter_nd_op},
        {"SegmentSum", translate_segment_sum_op},
        {"SparseToDense", translate_sparse_to_dense_op},
        {"Select", translate_select_op},
        {"SelectV2", translate_select_v2_op},
        {"Shape", translate_shape_op},
        {"Size", translate_size_op},
        {"Slice", translate_slice_op},
        {"Snapshot", translate_identity_op},
        {"Softmax", translate_softmax_op},
        {"SpaceToDepth", translate_space_to_depth_op},
        {"SparseReshape", translate_sparse_reshape_op},
        {"Split", translate_split_op},
        {"SplitV", translate_split_v_op},
        {"StopGradient", translate_identity_op},
        {"Sqrt", translate_sqrt_op},
        {"Square", translate_square_op},
        {"Squeeze", translate_squeeze_op},
        {"SpaceToBatchND", translate_space_to_batch_nd_op},
        {"StatefulPartitionedCall", translate_partitioned_call_op},
        {"StatelessIf", translate_if_op},
        {"StatelessWhile", translate_while_op},
        {"StridedSlice", translate_strided_slice_op},
        {"TensorListFromTensor", translate_tensor_list_from_tensor_op},
        {"TensorListGetItem", translate_tensor_list_get_item_op},
        {"TensorListPushBack", translate_tensor_list_push_back_op},
        {"TensorListSetItem", translate_tensor_list_set_item_op},
        {"TensorListStack", translate_tensor_list_stack_op},
        {"TensorListReserve", translate_tensor_list_reserve_op},
        {"Tile", translate_tile_op},
        {"TopK", translate_top_k_op},
        {"TopKV2", translate_top_k_v2_op},
        {"Transpose", translate_transpose_op},
        {"Unpack", translate_unpack_op},
        {"While", translate_while_op},
        {"Where", translate_where_op},
        {"Xdivy", translate_x_div_y_op},
        {"ZerosLike", translate_zeros_like_op},

        // Translators for internal operations
        {"BlockLSTM", translate_block_lstm_op},
        {"GRUBlockCell", translate_gru_block_cell_op},
        {"SparseFillEmptyRows", translate_sparse_fill_empty_rows_op},
        {"SparseSegmentSum", translate_sparse_segment_sum_op},
        {"Unique", translate_unique_op},
    };
};
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov