#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/script.h>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>

typedef torch::jit::script::Module ScriptModule;
#define DEFAULT_CUDA_DEVICE torch::Device(torch::kCUDA, 3)
#define CPU_DEVICE torch::Device(torch::kCPU)

#define NUM_EXPERTS 12
#define NUM_LAYERS 12

template <typename T>
struct DoNothingDeleter {
  void operator()(T* ptr) const {}
};

void
SetModuleContinuousMemory(torch::jit::script::Module* model)
{
  for (auto it = model->parameters().begin(); it != model->parameters().end();
       ++it) {
    if (!(*it).is_contiguous())
      (*it).set_data((*it).contiguous());
  }

  for (auto it = model->buffers().begin(); it != model->buffers().end(); ++it) {
    if (!(*it).is_contiguous())
      (*it).set_data((*it).contiguous());
  }
}

std::int64_t
GetModuleByteSize(torch::jit::script::Module* model)
{
  std::int64_t param_size = 0;
  for (auto it = model->parameters().begin(); it != model->parameters().end();
       ++it) {
    param_size += (*it).nbytes();
  }

  for (auto it = model->buffers().begin(); it != model->buffers().end(); ++it) {
    param_size += (*it).nbytes();
  }

  return param_size;
}

void
CopyModuleUnifiedMemory(torch::jit::script::Module* model, void* host_ptr)
{
  std::int64_t param_size = 0;
  for (auto it = model->parameters().begin(); it != model->parameters().end();
       ++it) {
    void* ptr = (*it).data_ptr();
    size_t size = (*it).nbytes();
    cudaMemcpy((char*)host_ptr + param_size, ptr, size, cudaMemcpyDefault);
    param_size += size;
  }

  for (auto it = model->buffers().begin(); it != model->buffers().end(); ++it) {
    void* ptr = (*it).data_ptr();
    size_t size = (*it).nbytes();
    cudaMemcpy((char*)host_ptr + param_size, ptr, size, cudaMemcpyDefault);
    param_size += size;
  }
}

#define FLOAT32_TENSOR_OPTIONS(target) \
  torch::TensorOptions().dtype(torch::kFloat32).device(target)

void
SetModulePinnedMemory(torch::jit::script::Module* model, void* host_ptr)
{
  std::int64_t param_size = 0;
  auto tensor_options = FLOAT32_TENSOR_OPTIONS(DEFAULT_CUDA_DEVICE);
  for (auto it = model->parameters().begin(); it != model->parameters().end();
       ++it) {
    size_t size = (*it).nbytes();
    (*it).set_data(torch::from_blob(
        (char*)host_ptr + param_size, (*it).sizes(), DoNothingDeleter<void>{},
        tensor_options));
    param_size += size;
  }

  for (auto it = model->buffers().begin(); it != model->buffers().end(); ++it) {
    size_t size = (*it).nbytes();
    (*it).set_data(torch::from_blob(
        (char*)host_ptr + param_size, (*it).sizes(), DoNothingDeleter<void>{},
        tensor_options));
    param_size += size;
  }
}

int
main()
{
  std::filesystem::path model_repo =
      "/mnt/raid0nvme1/xly/swap-engine/model_repo_switch-xxl-128";
  std::string model_name = "switch-xxl-128_decoder_expert_";

  cudaSetDevice(DEFAULT_CUDA_DEVICE.index());

  // print cuda free memory
  size_t free_byte;
  size_t total_byte;
  cudaMemGetInfo(&free_byte, &total_byte);
  std::cout << "free memory: " << free_byte / 1024 / 1024 << "MB" << std::endl;

  ScriptModule m =
      torch::jit::load(model_repo / (model_name + "1_0") / "0" / "model.pt");
  auto param_size = GetModuleByteSize(&m);

  std::cout << "param size: " << param_size / 1024 / 1024 << "MB" << std::endl;

  std::vector<ScriptModule*> modules;
  void* ptr = nullptr;
  cudaMallocManaged(&ptr, param_size * NUM_EXPERTS * NUM_LAYERS / 2);

  for (int i = 1; i < NUM_LAYERS; i += 2) {
    for (int j = 0; j < NUM_EXPERTS; j++) {
      auto module_path =
          model_repo /
          (model_name + std::to_string(i) + "_" + std::to_string(j)) / "0" /
          "model.pt";

      std::cout << "loading model: " << module_path << std::endl;
      ScriptModule* module = new ScriptModule(torch::jit::load(module_path));

      SetModuleContinuousMemory(module);
      CopyModuleUnifiedMemory(
          module, (char*)ptr + param_size * ((i - 1) / 2 * NUM_EXPERTS + j));
      SetModulePinnedMemory(
          module, (char*)ptr + param_size * ((i - 1) / 2 * NUM_EXPERTS + j));
      modules.push_back(module);
    }
  }

  cudaMemGetInfo(&free_byte, &total_byte);
  std::cout << "free memory: " << free_byte / 1024 / 1024 << "MB" << std::endl;

  auto input = torch::randn({5, 4096}, DEFAULT_CUDA_DEVICE);

  for (auto module : modules) {
    for (int i = 0; i < 10; i++) {
      auto start_time = std::chrono::system_clock::now();
      auto output = module->forward({input}).toTensor();
      auto end_time = std::chrono::system_clock::now();
      auto elapsed_microseconds =
          std::chrono::duration_cast<std::chrono::microseconds>(
              end_time - start_time);
      std::cout << "elapsed time: " << elapsed_microseconds.count() << "us"
                << std::endl;
    }
    cudaMemGetInfo(&free_byte, &total_byte);
    std::cout << "free memory: " << free_byte / 1024 / 1024 << "MB"
              << std::endl;
  }

  cudaMemGetInfo(&free_byte, &total_byte);
  std::cout << "free memory: " << free_byte / 1024 / 1024 << "MB" << std::endl;

  // clean up memory here
  for (auto module : modules) {
    delete module;
  }
  cudaFree(ptr);
}