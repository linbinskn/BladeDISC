/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/mlir/disc/tests/mlir_feature_test.h"
#include "tensorflow/compiler/mlir/disc/tests/mlir_test.h"
#include "tensorflow/core/platform/test.h"

namespace mlir_test {

const std::string c_ft_path =
    "tensorflow/compiler/mlir/disc/tests/tensorflow_ops/data/";

// Comment qgemm int8 GPU test since it only works when blade_gemm
// is on

TEST(TFQuantziedMatMul, PARTIAL_DYNAMIC_SHAPE_NHWC_I8_PER_CHANNEL) {
  setenv("DISC_ENABLE_COMPUTE_INTENSIVE_FUSE", "true", 1);
  // compute-intensive fusion should be used along with stitch fusion.
  setenv("DISC_ENABLE_STITCH", "true", 1);
  // TODO: this test fails when there is extra tail dynamic_broadcast_in_dim.
  // The shape constraint IR optimization helps to eliminate the bcasts. Thus
  // the environment of shape constraint IR is enabled currently. The bug will
  // be fixed in another PR.
  setenv("DISC_ENABLE_SHAPE_CONSTRAINT_IR", "true", 1);

  std::vector<float> inputs(32 * 64, 2.0);
  tensorflow::Tensor output(tensorflow::DataType::DT_FLOAT, {32, 128});
  auto datas = output.flat<float>();
  for (int i = 0; i < output.NumElements(); ++i) datas(i) = 128.0;
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "quantized_matmul_p_i8_per_tensor.mlir",
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"32x64xf32_X"},
      /*output_descriptors*/ {"f32_X"},
      /*input_vals*/ {inputs},
      /*expect_output_vals*/ {output}));

  unsetenv("DISC_ENABLE_SHAPE_CONSTRAINT_IR");
  unsetenv("DISC_ENABLE_STITCH");
  unsetenv("DISC_ENABLE_COMPUTE_INTENSIVE_FUSE");
}

}  // namespace mlir_test
