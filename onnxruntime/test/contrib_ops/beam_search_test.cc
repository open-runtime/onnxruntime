// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <vector>
#include "gtest/gtest.h"
#include "core/common/gsl.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

//TEST(BeamSearchTest, T5Op) {
//    std::cout << "starting t5 test" << std::endl;
//    OpTester test("BeamSearch", 1, kMSDomain);
//    test.AddInput<int32_t>("input_ids", {136}, {
//    1,   536,  9804,    67,   869,    67,  2730,    12,  1080,    16,   963,    33,
//    7036,  4672,   203,   565 ,  309 , 1549 ,   12 , 1080 ,   16 , 5252 , 4672 ,  203,
//    3639 ,  533  , 273  , 533  ,  18 , 3015 , 2668 , 3158 ,   17 ,   28 , 6134 ,  203,
//    3639 , 5690 ,  273 , 7354,    55, 20105  ,  18 ,21185 ,20105 , 6747  ,  12 ,23310,
//    4670 ,   18,    53,  8826 ,   12 , 1080,  3719 ,  203 ,  565 ,  309 ,  486 , 5690,
//    18, 26810 ,13332  , 203 , 3639 , 1002 , 2068 , 2668,  1941 ,11281 ,  501 , 1093,
//    13  , 203 ,  565  , 309  , 963  , 353 ,  599 ,   30  , 203 , 3639 ,  963  , 273,
//    5690  ,  18 , 1886 , 1225 , 1435  , 203 , 3639 , 1316 ,  273, 23425 ,   18  ,  53,
//    2040  ,  12,  1467 ,   16 ,23425  ,  18 ,   53  ,2040  ,  18 , 1630  ,  67  , 985,
//    5887 , 1578  ,  13 ,  203  ,3639, 21295 ,  273 ,23425 ,   18 ,   53 ,15775 ,   12,
//    2730 ,   13 ,  203 , 3639 ,5690 ,   18 , 5902  ,  12 ,   84, 11606 ,   13,   203,
//    565 ,  327 , 1316  ,   2});
//    test.AddInput<int64_t>("max_length", {1}, {50});
//    test.AddInput<int64_t>("min_length", {1}, {1});
//    test.AddInput<int32_t>("num_beams", {1}, {4});
//    test.AddInput<int32_t>("num_return_sequences", {1}, {1});
//    test.AddInput<float>("length_penalty", {1}, {1.5f});
//    test.AddInput<float>("repetition_penalty", {1}, {30.0f});
//    std::vector<int32_t> vocab_mask(32100, 1);
//    test.AddInput<int32_t>("vocab_mask", {32100}, vocab_mask);
//    test.AddOutput<int32_t>("sequences", {50}, {
//        0,     1,  2723,   279, 11281,   533,   358,   392,  1316,
//        18,   203,    36,   891,   533,  1021,  9804,   501,   261,
//        5159,    17,    28,  2934,   203,    36,   891,   963,  6321,
//        434,   326,  4957,   316,  8948,    16,   309,   486,  2112,
//        518,   903,   506,  1399,   487,   805,  1225,  1435,     0,
//        0,     0,     0,     0,     0});
//    std::vector<float> encoder_hidden_states(512, 1.0);
//    test.AddOutput<float>("sequences_scores", {1}, {-0.22555348});
//    test.AddOutput<float>("scores", {1}, {-0.22555348});
//    test.AddOutput<float>("encoder_hidden_states", {512}, encoder_hidden_states);
//    test.Run();
//}

TEST(BeamSearchTest, T5ForCodeDescription) {
  std::cout << "starting t5 test" << std::endl;
  std::vector<int64_t> input_ids_shape{1, 136};
    std::vector<int32_t> input_ids{
            1,   536,  9804,    67,   869,    67,  2730,    12,  1080,    16,   963,    33,
            7036,  4672,   203,   565 ,  309 , 1549 ,   12 , 1080 ,   16 , 5252 , 4672 ,  203,
            3639 ,  533  , 273  , 533  ,  18 , 3015 , 2668 , 3158 ,   17 ,   28 , 6134 ,  203,
            3639 , 5690 ,  273 , 7354,    55, 20105  ,  18 ,21185 ,20105 , 6747  ,  12 ,23310,
            4670 ,   18,    53,  8826 ,   12 , 1080,  3719 ,  203 ,  565 ,  309 ,  486 , 5690,
            18, 26810 ,13332  , 203 , 3639 , 1002 , 2068 , 2668,  1941 ,11281 ,  501 , 1093,
            13  , 203 ,  565  , 309  , 963  , 353 ,  599 ,   30  , 203 , 3639 ,  963  , 273,
            5690  ,  18 , 1886 , 1225 , 1435  , 203 , 3639 , 1316 ,  273, 23425 ,   18  ,  53,
            2040  ,  12,  1467 ,   16 ,23425  ,  18 ,   53  ,2040  ,  18 , 1630  ,  67  , 985,
            5887 , 1578  ,  13 ,  203  ,3639, 21295 ,  273 ,23425 ,   18 ,   53 ,15775 ,   12,
            2730 ,   13 ,  203 , 3639 ,5690 ,   18 , 5902  ,  12 ,   84, 11606 ,   13,   203,
            565 ,  327 , 1316  ,   2};

  std::vector<int64_t> parameter_shape{1};
  std::vector<int32_t> max_length{50};
  std::vector<int32_t> min_length{1};
  std::vector<int32_t> num_beams{4};
  std::vector<int32_t> num_return_sequences{1};
  std::vector<float> length_penalty{1.5f};
  std::vector<float> repetition_penalty{30.0f};
  std::vector<int32_t> vocab_mask(32100, 1);

  std::vector<int64_t> expected_output_shape{1, 50};
  std::vector<int32_t> expected_output{
          0,     1,  2723,   279, 11281,   533,   358,   392,  1316,
          18,   203,    36,   891,   533,  1021,  9804,   501,   261,
          5159,    17,    28,  2934,   203,    36,   891,   963,  6321,
          434,   326,  4957,   316,  8948,    16,   309,   486,  2112,
          518,   903,   506,  1399,   487,   805,  1225,  1435,     0,
          0,     0,     0,     0,     0};

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  auto input_ids_tensor = Ort::Value::CreateTensor(
      info, input_ids.data(), input_ids.size(), input_ids_shape.data(), input_ids_shape.size());

  auto max_length_tensor = Ort::Value::CreateTensor(
      info, max_length.data(), max_length.size(), parameter_shape.data(), parameter_shape.size());

  auto min_length_tensor = Ort::Value::CreateTensor(
      info, min_length.data(), min_length.size(), parameter_shape.data(), parameter_shape.size());

  auto num_beams_tensor = Ort::Value::CreateTensor(
      info, num_beams.data(), num_beams.size(), parameter_shape.data(), parameter_shape.size());

  auto num_return_sequences_tensor = Ort::Value::CreateTensor(
      info, num_return_sequences.data(), num_return_sequences.size(), parameter_shape.data(), parameter_shape.size());

  auto length_penalty_tensor = Ort::Value::CreateTensor(
      info, length_penalty.data(), length_penalty.size(), parameter_shape.data(), parameter_shape.size());

  auto repetition_penalty_tensor = Ort::Value::CreateTensor(
      info, repetition_penalty.data(), repetition_penalty.size(), parameter_shape.data(), parameter_shape.size());

  std::vector<Ort::Value> ort_inputs;
  ort_inputs.push_back(std::move(input_ids_tensor));
  ort_inputs.push_back(std::move(max_length_tensor));
  ort_inputs.push_back(std::move(min_length_tensor));
  ort_inputs.push_back(std::move(num_beams_tensor));
  ort_inputs.push_back(std::move(num_return_sequences_tensor));
  ort_inputs.push_back(std::move(length_penalty_tensor));
  ort_inputs.push_back(std::move(repetition_penalty_tensor));
  const char* input_names[] = {"input_ids", "max_length", "min_length", "num_beams", "num_return_sequences",
                               "length_penalty", "repetition_penalty", "vocab_mask"};
  const char* const output_names[] = {"sequences", "sequence_scores", "scores", "encoder_hidden_states"};

  Ort::SessionOptions session_options;
#ifdef USE_CUDA
  Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
#endif

  Ort::Session session(*ort_env, ORT_TSTR("testdata/transformers/t5_small_beam_search.onnx"), session_options);
  auto ort_outputs = session.Run(Ort::RunOptions{}, input_names, ort_inputs.data(), ort_inputs.size(),
                                 output_names, 4);

//  ASSERT_EQ(ort_outputs.size(), 1U);
  const auto& sequences = ort_outputs[0];
  const auto& encoder_hidden_states = ort_outputs[3];
  std::cout << "encoder_hidden_states: " << encoder_hidden_states << std::endl;
  ASSERT_TRUE(sequences.IsTensor());

  auto result_ts = sequences.GetTensorTypeAndShapeInfo();
  ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, result_ts.GetElementType());

  ASSERT_EQ(expected_output_shape, result_ts.GetShape());
  const auto* result_vals = sequences.GetTensorData<int32_t>();
  auto result_span = gsl::make_span(result_vals, expected_output.size());
  ASSERT_TRUE(std::equal(expected_output.cbegin(), expected_output.cend(), result_span.begin(), result_span.end()));
}

//TEST(BeamSearchTest, GptBeamSearchFp32) {
//  std::vector<int64_t> input_ids_shape{3, 12};
//  std::vector<int32_t> input_ids{
//      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620,
//      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572,
//      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328};
//
//  std::vector<int64_t> parameter_shape{1};
//  std::vector<int32_t> max_length{20};
//  std::vector<int32_t> min_length{1};
//  std::vector<int32_t> num_beams{4};
//  std::vector<int32_t> num_return_sequences{1};
//  std::vector<float> length_penalty{1.0f};
//  std::vector<float> repetition_penalty{1.0f};
//
//  std::vector<int64_t> expected_output_shape{input_ids_shape[0], num_return_sequences[0], max_length[0]};
//  std::vector<int32_t> expected_output{
//      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620, 131, 131, 131, 181, 638, 638, 638, 638,
//      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572, 292, 292, 292, 292, 292, 292, 292, 292,
//      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328, 328, 669, 669, 669, 669, 669, 669, 669};
//
//  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
//  auto input_ids_tensor = Ort::Value::CreateTensor(
//      info, input_ids.data(), input_ids.size(), input_ids_shape.data(), input_ids_shape.size());
//
//  auto max_length_tensor = Ort::Value::CreateTensor(
//      info, max_length.data(), max_length.size(), parameter_shape.data(), parameter_shape.size());
//
//  auto min_length_tensor = Ort::Value::CreateTensor(
//      info, min_length.data(), min_length.size(), parameter_shape.data(), parameter_shape.size());
//
//  auto num_beams_tensor = Ort::Value::CreateTensor(
//      info, num_beams.data(), num_beams.size(), parameter_shape.data(), parameter_shape.size());
//
//  auto num_return_sequences_tensor = Ort::Value::CreateTensor(
//      info, num_return_sequences.data(), num_return_sequences.size(), parameter_shape.data(), parameter_shape.size());
//
//  auto length_penalty_tensor = Ort::Value::CreateTensor(
//      info, length_penalty.data(), length_penalty.size(), parameter_shape.data(), parameter_shape.size());
//
//  auto repetition_penalty_tensor = Ort::Value::CreateTensor(
//      info, repetition_penalty.data(), repetition_penalty.size(), parameter_shape.data(), parameter_shape.size());
//
//  std::vector<Ort::Value> ort_inputs;
//  ort_inputs.push_back(std::move(input_ids_tensor));
//  ort_inputs.push_back(std::move(max_length_tensor));
//  ort_inputs.push_back(std::move(min_length_tensor));
//  ort_inputs.push_back(std::move(num_beams_tensor));
//  ort_inputs.push_back(std::move(num_return_sequences_tensor));
//  ort_inputs.push_back(std::move(length_penalty_tensor));
//  ort_inputs.push_back(std::move(repetition_penalty_tensor));
//  const char* input_names[] = {"input_ids", "max_length", "min_length", "num_beams", "num_return_sequences",
//                               "length_penalty", "repetition_penalty"};
//  const char* const output_names[] = {"sequences"};
//
//  Ort::SessionOptions session_options;
//#ifdef USE_CUDA
//  Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
//#endif
//
//  Ort::Session session(*ort_env, ORT_TSTR("testdata/transformers/tiny_gpt2_beamsearch.onnx"), session_options);
//  auto ort_outputs = session.Run(Ort::RunOptions{}, input_names, ort_inputs.data(), ort_inputs.size(),
//                                 output_names, 1);
//
//  ASSERT_EQ(ort_outputs.size(), 1U);
//  const auto& sequences = ort_outputs[0];
//  const auto& encoder_hidden_states = ort_outputs[1];
//  std::cout << "encoder_hidden_states: " << encoder_hidden_states << std::endl;
//  ASSERT_TRUE(sequences.IsTensor());
//
//  auto result_ts = sequences.GetTensorTypeAndShapeInfo();
//  ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, result_ts.GetElementType());
//
//  ASSERT_EQ(expected_output_shape, result_ts.GetShape());
//  const auto* result_vals = sequences.GetTensorData<int32_t>();
//  auto result_span = gsl::make_span(result_vals, expected_output.size());
//  ASSERT_TRUE(std::equal(expected_output.cbegin(), expected_output.cend(), result_span.begin(), result_span.end()));
//}
//
//TEST(BeamSearchTest, GptBeamSearchFp16) {
//  std::vector<int64_t> input_ids_shape{3, 12};
//  std::vector<int32_t> input_ids{
//      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620,
//      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572,
//      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328};
//
//  std::vector<int64_t> parameter_shape{1};
//  std::vector<int32_t> max_length{20};
//  std::vector<int32_t> min_length{1};
//  std::vector<int32_t> num_beams{4};
//  std::vector<int32_t> num_return_sequences{1};
//  std::vector<float> length_penalty{1.0f};
//  std::vector<float> repetition_penalty{1.0f};
//
//  std::vector<int64_t> expected_output_shape{input_ids_shape[0], num_return_sequences[0], max_length[0]};
//
//  std::vector<int32_t> expected_output{
//      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620, 131, 131, 131, 181, 638, 638, 638, 638,
//      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572, 292, 292, 292, 292, 292, 292, 292, 292,
//      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328, 328, 669, 669, 669, 669, 669, 669, 669};
//
//  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
//  auto input_ids_tensor = Ort::Value::CreateTensor(
//      info, input_ids.data(), input_ids.size(), input_ids_shape.data(), input_ids_shape.size());
//
//  auto max_length_tensor = Ort::Value::CreateTensor(
//      info, max_length.data(), max_length.size(), parameter_shape.data(), parameter_shape.size());
//
//  auto min_length_tensor = Ort::Value::CreateTensor(
//      info, min_length.data(), min_length.size(), parameter_shape.data(), parameter_shape.size());
//
//  auto num_beams_tensor = Ort::Value::CreateTensor(
//      info, num_beams.data(), num_beams.size(), parameter_shape.data(), parameter_shape.size());
//
//  auto num_return_sequences_tensor = Ort::Value::CreateTensor(
//      info, num_return_sequences.data(), num_return_sequences.size(), parameter_shape.data(), parameter_shape.size());
//
//  auto length_penalty_tensor = Ort::Value::CreateTensor(
//      info, length_penalty.data(), length_penalty.size(), parameter_shape.data(), parameter_shape.size());
//
//  auto repetition_penalty_tensor = Ort::Value::CreateTensor(
//      info, repetition_penalty.data(), repetition_penalty.size(), parameter_shape.data(), parameter_shape.size());
//
//  std::vector<Ort::Value> ort_inputs;
//  ort_inputs.push_back(std::move(input_ids_tensor));
//  ort_inputs.push_back(std::move(max_length_tensor));
//  ort_inputs.push_back(std::move(min_length_tensor));
//  ort_inputs.push_back(std::move(num_beams_tensor));
//  ort_inputs.push_back(std::move(num_return_sequences_tensor));
//  ort_inputs.push_back(std::move(length_penalty_tensor));
//  ort_inputs.push_back(std::move(repetition_penalty_tensor));
//  const char* input_names[] = {"input_ids", "max_length", "min_length", "num_beams", "num_return_sequences",
//                               "length_penalty", "repetition_penalty"};
//  const char* const output_names[] = {"sequences"};
//
//  constexpr int min_cuda_architecture = 530;
//  if (HasCudaEnvironment(min_cuda_architecture)) {
//    Ort::SessionOptions session_options;
//#ifdef USE_CUDA
//    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
//#endif
//
//    // The ONNX model is generated like the following:
//    // python convert_generation.py --model_type gpt2 -m hf-internal-testing/tiny-random-gpt2
//    //        --output tiny_gpt2_beamsearch_fp16.onnx  -p fp16 --use_gpu --max_length 20
//    Ort::Session session(*ort_env, ORT_TSTR("testdata/transformers/tiny_gpt2_beamsearch_fp16.onnx"), session_options);
//
//    auto ort_outputs = session.Run(Ort::RunOptions{}, input_names, ort_inputs.data(), ort_inputs.size(),
//                                   output_names, 1);
//
//    ASSERT_EQ(ort_outputs.size(), 1U);
//    const auto& sequences = ort_outputs[0];
//    ASSERT_TRUE(sequences.IsTensor());
//
//    auto result_ts = sequences.GetTensorTypeAndShapeInfo();
//    ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, result_ts.GetElementType());
//
//    ASSERT_EQ(expected_output_shape, result_ts.GetShape());
//    const auto* result_vals = sequences.GetTensorData<int32_t>();
//    auto result_span = gsl::make_span(result_vals, expected_output.size());
//    ASSERT_TRUE(std::equal(expected_output.cbegin(), expected_output.cend(), result_span.begin(), result_span.end()));
//  }
//}

//TEST(BeamSearchTest, GptBeamSearchFp16_VocabPadded) {
//  std::vector<int64_t> input_ids_shape{3, 12};
//  std::vector<int32_t> input_ids{
//      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620,
//      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572,
//      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328};
//
//  std::vector<int64_t> parameter_shape{1};
//  std::vector<int32_t> max_length{20};
//  std::vector<int32_t> min_length{1};
//  std::vector<int32_t> num_beams{4};
//  std::vector<int32_t> num_return_sequences{1};
//  std::vector<float> length_penalty{1.0f};
//  std::vector<float> repetition_penalty{1.0f};
//
//  std::vector<int64_t> expected_output_shape{input_ids_shape[0], num_return_sequences[0], max_length[0]};
//
//  std::vector<int32_t> expected_output{
//      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620, 131, 131, 131, 181, 638, 638, 638, 638,
//      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572, 292, 292, 292, 292, 292, 292, 292, 292,
//      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328, 328, 669, 669, 669, 669, 669, 669, 669};
//
//  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
//  auto input_ids_tensor = Ort::Value::CreateTensor(
//      info, input_ids.data(), input_ids.size(), input_ids_shape.data(), input_ids_shape.size());
//
//  auto max_length_tensor = Ort::Value::CreateTensor(
//      info, max_length.data(), max_length.size(), parameter_shape.data(), parameter_shape.size());
//
//  auto min_length_tensor = Ort::Value::CreateTensor(
//      info, min_length.data(), min_length.size(), parameter_shape.data(), parameter_shape.size());
//
//  auto num_beams_tensor = Ort::Value::CreateTensor(
//      info, num_beams.data(), num_beams.size(), parameter_shape.data(), parameter_shape.size());
//
//  auto num_return_sequences_tensor = Ort::Value::CreateTensor(
//      info, num_return_sequences.data(), num_return_sequences.size(), parameter_shape.data(), parameter_shape.size());
//
//  auto length_penalty_tensor = Ort::Value::CreateTensor(
//      info, length_penalty.data(), length_penalty.size(), parameter_shape.data(), parameter_shape.size());
//
//  auto repetition_penalty_tensor = Ort::Value::CreateTensor(
//      info, repetition_penalty.data(), repetition_penalty.size(), parameter_shape.data(), parameter_shape.size());
//
//  std::vector<Ort::Value> ort_inputs;
//  ort_inputs.push_back(std::move(input_ids_tensor));
//  ort_inputs.push_back(std::move(max_length_tensor));
//  ort_inputs.push_back(std::move(min_length_tensor));
//  ort_inputs.push_back(std::move(num_beams_tensor));
//  ort_inputs.push_back(std::move(num_return_sequences_tensor));
//  ort_inputs.push_back(std::move(length_penalty_tensor));
//  ort_inputs.push_back(std::move(repetition_penalty_tensor));
//  const char* input_names[] = {"input_ids", "max_length", "min_length", "num_beams", "num_return_sequences",
//                               "length_penalty", "repetition_penalty"};
//  const char* const output_names[] = {"sequences"};
//
//  constexpr int min_cuda_architecture = 530;
//  if (HasCudaEnvironment(min_cuda_architecture)) {
//    Ort::SessionOptions session_options;
//#ifdef USE_CUDA
//    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
//#endif
//
//    // The following model was obtained by padding the vocabulary size in testdata/transformers/tiny_gpt2_beamsearch_fp16.onnx
//    // from 1000 to 1600 (just for illustrative and testing purposes) to see if the beam search implementation can handle
//    // such a scenario
//    Ort::Session session(*ort_env, ORT_TSTR("testdata/transformers/tiny_gpt2_beamsearch_fp16_padded_vocab.onnx"), session_options);
//
//    auto ort_outputs = session.Run(Ort::RunOptions{}, input_names, ort_inputs.data(), ort_inputs.size(),
//                                   output_names, 1);
//
//    ASSERT_EQ(ort_outputs.size(), 1U);
//    const auto& sequences = ort_outputs[0];
//    ASSERT_TRUE(sequences.IsTensor());
//
//    auto result_ts = sequences.GetTensorTypeAndShapeInfo();
//    ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, result_ts.GetElementType());
//
//    ASSERT_EQ(expected_output_shape, result_ts.GetShape());
//    const auto* result_vals = sequences.GetTensorData<int32_t>();
//    auto result_span = gsl::make_span(result_vals, expected_output.size());
//    ASSERT_TRUE(std::equal(expected_output.cbegin(), expected_output.cend(), result_span.begin(), result_span.end()));
//  }
//}
}  // namespace test
}  // namespace onnxruntime
