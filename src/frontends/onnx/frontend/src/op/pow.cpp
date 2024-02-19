// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/pow.hpp"

#include "openvino/frontend/exception.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/power.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector pow(const ov::frontend::onnx::Node& node) {
    auto inputs = node.get_ov_inputs();
    FRONT_END_GENERAL_CHECK(inputs.size() == 2, "Power operation requires 2 inputs. Got: ", inputs.size());

    auto base = inputs[0];
    auto exponent = inputs[1];
    auto base_type = inputs[0].get_element_type();
    auto exponent_type = inputs[1].get_element_type();
    if (exponent_type != base_type) {
        if (exponent_type.is_integral() || (base_type.is_real() && base_type.bitwidth() >= exponent_type.bitwidth())) {
            exponent = std::make_shared<v0::Convert>(exponent, base_type);
        } else {
            base = std::make_shared<v0::Convert>(base, exponent_type);
            auto power = std::make_shared<v1::Power>(base, exponent);
            return {std::make_shared<v0::Convert>(power, base_type)};
        }
    }
    return {std::make_shared<v1::Power>(base, exponent)};
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
