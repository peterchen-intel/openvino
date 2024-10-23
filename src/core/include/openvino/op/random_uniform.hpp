// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {
namespace op {
namespace v8 {
/// \brief Tensor RandomUniform operation.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API RandomUniform : public Op {
public:
    OPENVINO_OP("RandomUniform", "opset8");

    RandomUniform() = default;

    /// \brief      Constructs a RandomUniform operation.
    ///
    /// \param      out_shape         Node producing the tensor with output shape.
    /// \param      min_val           Node producing the tensor with minimum value.
    /// \param      max_val           Node producing the tensor with maximum value.
    /// \param      out_type          Output type of the tensor.
    /// \param      global_seed       Global seed value.
    /// \param      op_seed           Operational seed value.
    /// \param      alignment         Alignment of numbers generated by Philox algorithm based on provided seed.
    RandomUniform(const Output<Node>& out_shape,
                  const Output<Node>& min_val,
                  const Output<Node>& max_val,
                  const ov::element::Type& out_type,
                  uint64_t global_seed = 0,
                  uint64_t op_seed = 0,
                  ov::op::PhiloxAlignment alignment = ov::op::PhiloxAlignment::TENSORFLOW);

    void validate_and_infer_types() override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    /// \return Turns off constant folding for RandomUniform operation.
    bool can_constant_fold(const OutputVector& inputs_values) const override;

    /// \return The output tensor type.
    const ov::element::Type& get_out_type() const;

    void set_out_type(const ov::element::Type& output_type);

    /// \return The global seed value.
    uint64_t get_global_seed() const;

    void set_global_seed(uint64_t seed);

    /// \return The operational seed value.
    uint64_t get_op_seed() const;

    void set_op_seed(uint64_t seed2);

    /// \return The state value.
    std::pair<uint64_t, uint64_t> get_state() const;

    /// \return The alignment mode.
    ov::op::PhiloxAlignment get_alignment() const;

    void set_alignment(ov::op::PhiloxAlignment alignmnent);

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;

protected:
    ov::element::Type m_output_type;
    uint64_t m_global_seed;
    uint64_t m_op_seed;
    ov::op::PhiloxAlignment m_alignment = ov::op::PhiloxAlignment::TENSORFLOW;

    mutable std::pair<uint64_t, uint64_t> m_state;
};
}  // namespace v8
}  // namespace op
}  // namespace ov
