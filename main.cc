#include <gflags/gflags.h>
// #include <onnxruntime_c_api.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <onnxruntime_cxx_api.h>
#include <unistd.h>
#include <string.h>

#include <iostream>

// #define BUFFERSIZE 104857600

struct CudaMemoryDeleter {
  explicit CudaMemoryDeleter(const Ort::Allocator* alloc) { alloc_ = alloc; }

  void operator()(void* ptr) const { alloc_->Free(ptr); }

  const Ort::Allocator* alloc_;
};

int main() {
  

  // std::cout << "Running inference..." << std::endl;
  // return 0;

  const auto& api = Ort::GetApi();

  auto providers = Ort::GetAvailableProviders();
  for (const auto& provider : providers) {
    std::cout << provider << std::endl;
  }
  // return 0;

  // // Enable cuda graph in cuda provider option.
  // OrtCUDAProviderOptionsV2* cuda_options = nullptr;
  // api.CreateCUDAProviderOptions(&cuda_options);
  // std::unique_ptr<OrtCUDAProviderOptionsV2,
  //                 decltype(api.ReleaseCUDAProviderOptions)>
  //     rel_cuda_options(cuda_options, api.ReleaseCUDAProviderOptions);
  // std::vector<const char*> keys{"enable_cuda_graph"};
  // std::vector<const char*> values{"1"};
  // api.UpdateCUDAProviderOptions(rel_cuda_options.get(), keys.data(),
  //                               values.data(), 1);

  Ort::SessionOptions session_options;
  // session_options.AppendExecutionProvider_CUDA_V2(*rel_cuda_options);
  Ort::ThrowOnError(
      OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
  // Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA_V2(session_options,
  // 0));
  //   api.SessionOptionsAppendExecutionProvider_CUDA_V2(
  //       static_cast<OrtSessionOptions*>(session_options),
  //       rel_cuda_options.get());

  OrtThreadingOptions* thread_options = nullptr;
  api.CreateThreadingOptions(&thread_options);

  Ort::Env ort_env(thread_options, ORT_LOGGING_LEVEL_VERBOSE, "Default");
  // Ort::Env ort_env(ORT_LOGGING_LEVEL_VERBOSE, "Default");

  char* buffer = (char*)malloc(104857600 * sizeof(char));

  FILE* filp = fopen("/home/xly/swap-engine/torch_model.onnx", "rb");
  int bytes_read = fread(buffer, sizeof(char), 104857600, filp);

  char* cuda_buffer = nullptr;
  cudaMalloc((void**)&cuda_buffer, 104857600 * sizeof(char));

  cudaMemcpy(cuda_buffer, buffer, 104857600 * sizeof(char),
             cudaMemcpyHostToDevice);

  // Create IO bound inputs and outputs.
  // Ort::Session session(ort_env, "/home/xly/swap-engine/torch_model.onnx",
  //                      session_options);

  std::cout << "before" << std::endl;
  sleep(5);

  session_options.SetExecutionMode(ORT_SEQUENTIAL);
  session_options.DisablePerSessionThreads();
  session_options.DisableProfiling();
  session_options.DisableMemPattern();

  Ort::Session session(ort_env, cuda_buffer, bytes_read, session_options);
  std::cout << "after" << std::endl;
  sleep(5);
  

  Ort::MemoryInfo info_cuda("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0,
                            OrtMemTypeDefault);

  Ort::Allocator cuda_allocator(session, info_cuda);

  const std::array<int64_t, 2> x_shape = {3, 2};
  std::array<float, 3 * 2> x_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto input_data = std::unique_ptr<void, CudaMemoryDeleter>(
      cuda_allocator.Alloc(x_values.size() * sizeof(float)),
      CudaMemoryDeleter(&cuda_allocator));
  cudaMemcpy(input_data.get(), x_values.data(), sizeof(float) * x_values.size(),
             cudaMemcpyHostToDevice);

  // Create an OrtValue tensor backed by data on CUDA memory
  Ort::Value bound_x = Ort::Value::CreateTensor(
      info_cuda, reinterpret_cast<float*>(input_data.get()), x_values.size(),
      x_shape.data(), x_shape.size());

  const std::array<int64_t, 2> expected_y_shape = {3, 2};
  std::array<float, 3 * 2> expected_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};
  auto output_data = std::unique_ptr<void, CudaMemoryDeleter>(
      cuda_allocator.Alloc(expected_y.size() * sizeof(float)),
      CudaMemoryDeleter(&cuda_allocator));

  // Create an OrtValue tensor backed by data on CUDA memory
  Ort::Value bound_y = Ort::Value::CreateTensor(
      info_cuda, reinterpret_cast<float*>(output_data.get()), expected_y.size(),
      expected_y_shape.data(), expected_y_shape.size());

  Ort::IoBinding binding(session);
  binding.BindInput("x_in", bound_x);
  binding.BindOutput("x_out", bound_y);

  // One regular run for necessary memory allocation and graph capturing
  session.Run(Ort::RunOptions(), binding);

  // After capturing, CUDA graph replay happens from this Run onwards
  session.Run(Ort::RunOptions(), binding);

  // Update input and then replay CUDA graph with the updated input
  x_values = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
  cudaMemcpy(input_data.get(), x_values.data(), sizeof(float) * x_values.size(),
             cudaMemcpyHostToDevice);
  session.Run(Ort::RunOptions(), binding);

  // cudaFree(cuda_buffer);
  free(buffer);

  return 0;
}