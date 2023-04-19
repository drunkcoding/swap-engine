#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/script.h>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>

#include "memory_pool.h"

template <typename T>
struct DoNothingDeleter {
  void operator()(T* ptr) const {}
};

CudaStreamPool* kCudaStreamPool = CudaStreamPool::GetInstance();

typedef torch::jit::script::Module ScriptModule;
#define DEFAULT_CUDA_DEVICE torch::Device(torch::kCUDA, 0)
#define CPU_DEVICE torch::Device(torch::kCPU)
#define CUDA_DEVICE(device_id) torch::Device(torch::kCUDA, device_id)
#define DISK_DEVICE(device_id) torch::Device(torch::kLazy)

#define FLOAT32_TENSOR_OPTIONS(target) \
  torch::TensorOptions().dtype(torch::kFloat32).device(target)
#define FAKE_TENSOR_SIZES torch::IntArrayRef({1})

#define MB (1024 * 1024)

// class ModuleFetch {
//  public:
//   ModuleFetch(const std::string& model_path)
//       : model_path_(model_path), model_(nullptr), model_copy_(nullptr),
//         device_(DISK_DEVICE)
//   {
//     // all initialize to empty model
//     model_ = new ScriptModule(torch::jit::load(model_path_));
//     SetModuleEmptyMemory(CPU_DEVICE);
//     model_copy_ = new ScriptModule((*model_).clone());
//   }
//   ~ModuleFetch() = default;

//   ScriptModule* Instance()
//   {
//     if (device_.is_cuda())
//       return model_copy_;
//     else
//       return model_;
//   }
//   void SetDevice(const torch::Device& device) { model_->to(device); }

//  private:
//   void SetModuleEmptyMemory(const torch::Device& device)
//   {
//     auto tensor_options = FLOAT32_TENSOR_OPTIONS(device);
//     for (auto it = model_->parameters().begin();
//          it != model_->parameters().end(); ++it) {
//       (*it).set_data(torch::empty(FAKE_TENSOR_SIZES, tensor_options));
//     }

//     for (auto it = model_->buffers().begin(); it != model_->buffers().end();
//          ++it) {
//       (*it).set_data(torch::empty(FAKE_TENSOR_SIZES, tensor_options));
//     }
//   }

//   void SetModuleContinuousMemory()
//   {
//     for (auto it = model_->parameters().begin(); it !=
//     model_parameters().end();
//          ++it) {
//       (*it).set_data((*it).contiguous());
//     }

//     for (auto it = model_buffers().begin(); it != model_buffers().end();
//     ++it) {
//       (*it).set_data((*it).contiguous());
//     }
//   }

//   void SetDisk()
//   {
//     cpu_params_.clear();
//     cpu_buffers_.clear();
//     gpu_params_.clear();
//     gpu_buffers_.clear();
//     SetModuleEmptyMemory(device_);
//     device_ = DISK_DEVICE;
//   }

//   void SetHost()
//   {
//     if (device_.is_cuda()) {
//     }

//     if (device_.is_lazy()) {
//       model_ = new ScriptModule(torch::jit::load(model_path_));
//       SetModuleContinuousMemory();
//     }
//     cpu_params_.clear();
//     cpu_buffers_.clear();
//     gpu_params_.clear();
//     gpu_buffers_.clear();
//     SetModuleEmptyMemory(device_);
//     device_ = CPU_DEVICE;
//   }

//  private:
//   std::string model_path_;
//   ScriptModule* model_;
//   ScriptModule* model_copy_;
//   torch::Device device_;
//   std::vector<at::Tensor> cpu_params_;
//   std::vector<at::Tensor> cpu_buffers_;
//   std::vector<at::Tensor> gpu_params_;
//   std::vector<at::Tensor> gpu_buffers_;
// };

inline std::size_t
GetFreeDeviceMemory(int device_id)
{
  size_t free_memory, total_memory;
  cudaSetDevice(device_id);
  cudaMemGetInfo(&free_memory, &total_memory);
  return free_memory;
}

void
SetModuleContinuousMemory(torch::jit::script::Module* model)
{
  for (auto it = model->parameters().begin(); it != model->parameters().end();
       ++it) {
    (*it).set_data((*it).contiguous());
  }

  for (auto it = model->buffers().begin(); it != model->buffers().end(); ++it) {
    (*it).set_data((*it).contiguous());
  }
}

void
SetModuleEmptyMemory(torch::jit::script::Module* model)
{
  auto tensor_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(CPU_DEVICE);
  auto tensor_sizes = torch::IntArrayRef({1});
  for (auto it = model->parameters().begin(); it != model->parameters().end();
       ++it) {
    (*it).set_data(torch::empty(tensor_sizes, tensor_options));
  }

  for (auto it = model->buffers().begin(); it != model->buffers().end(); ++it) {
    (*it).set_data(torch::empty(tensor_sizes, tensor_options));
  }
}

void
SetModuleCudaMemory(
    torch::jit::script::Module* model, void* device_ptr,
    const torch::Device& device)
{
  std::int64_t param_size = 0;
  auto tensor_options = FLOAT32_TENSOR_OPTIONS(device);
  std::cout << "tensor_options: " << tensor_options << std::endl;
  for (auto it = model->parameters().begin(); it != model->parameters().end();
       ++it) {
    size_t size = (*it).nbytes();
    (*it).set_data(torch::from_blob(
        (char*)device_ptr + param_size, (*it).sizes(), DoNothingDeleter<void>{},
        tensor_options));
    // std::cout << "param: " << (*it).device() << std::endl;
    param_size += size;
  }

  for (auto it = model->buffers().begin(); it != model->buffers().end(); ++it) {
    size_t size = (*it).nbytes();
    (*it).set_data(torch::from_blob(
        (char*)device_ptr + param_size, (*it).sizes(), DoNothingDeleter<void>{},
        tensor_options));
    // std::cout << "buffer: " << (*it).device() << std::endl;
    param_size += size;
  }
  // model->to(device);
}

void
CopyModulePinnedMemory(torch::jit::script::Module* model, void* host_ptr)
{
  std::int64_t param_size = 0;
  for (auto it = model->parameters().begin(); it != model->parameters().end();
       ++it) {
    void* ptr = (*it).data_ptr();
    size_t size = (*it).nbytes();
    memcpy((char*)host_ptr + param_size, ptr, size);
    param_size += size;
  }

  for (auto it = model->buffers().begin(); it != model->buffers().end(); ++it) {
    void* ptr = (*it).data_ptr();
    size_t size = (*it).nbytes();
    memcpy((char*)host_ptr + param_size, ptr, size);
    param_size += size;
  }
}

void
SetModulePinnedMemory(torch::jit::script::Module* model, void* host_ptr)
{
  std::int64_t param_size = 0;
  auto tensor_options = FLOAT32_TENSOR_OPTIONS(CPU_DEVICE);
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
  // model->to(CPU_DEVICE);
}

void
inference_cycle(
    std::string model_path, HostMemoryPool* host_memory_pool,
    DeviceMemoryPool* device_memory_pool, int id)
{
  torch::InferenceMode infer_guard(true);
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::ones({1, 3, 224, 224}));

  torch::jit::script::Module module;
  module = torch::jit::load(model_path);

  int param_byte_size = 0;
  for (const auto& param : module.parameters()) {
    param_byte_size += param.nbytes();
  }
  std::cout << "param_byte_size: " << param_byte_size << std::endl;

  int buffer_byte_size = 0;
  for (const auto& buffer : module.buffers()) {
    buffer_byte_size += buffer.nbytes();
  }
  std::cout << "buffer_byte_size: " << buffer_byte_size << std::endl;

  at::Tensor output = module.forward(inputs).toTensor();
  // std::cout << "output original: " << output << std::endl;

  void* host_ptr = host_memory_pool->AllocateMemory(
      id, param_byte_size + buffer_byte_size, CPU_DEVICE);

  SetModuleContinuousMemory(&module);
  CopyModulePinnedMemory(&module, host_ptr);
  SetModulePinnedMemory(&module, host_ptr);

  std::cout << "cuda Free Memory before CPU forward: " << GetFreeDeviceMemory(0)
            << std::endl;

  // Execute the model and turn its output into a tensor.
  auto start_time = std::chrono::system_clock::now();
  output = module.forward(inputs).toTensor();
  // std::cout << "output pinned: " << output << std::endl;
  std::cout << "cuda Free Memory after CPU forward: " << GetFreeDeviceMemory(0)
            << std::endl;

  auto end_time = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end_time - start_time;
  std::cout << "elapsed time: " << elapsed_seconds.count() << "s" << std::endl;

  // auto model_gpu = module;
  void* device_ptr = device_memory_pool->AllocateMemory(
      id, param_byte_size + buffer_byte_size, DEFAULT_CUDA_DEVICE);
  std::cout << "cuda Free Memory after Pool Allocation: "
            << GetFreeDeviceMemory(0) << std::endl;

  cudaMemcpy(
      device_ptr, host_ptr, param_byte_size + buffer_byte_size,
      cudaMemcpyHostToDevice);

  std::cout << "cuda Free Memory after Pool Copy: " << GetFreeDeviceMemory(0)
            << std::endl;


  SetModuleCudaMemory(&module, device_ptr, DEFAULT_CUDA_DEVICE);
  std::cout << "module: " << module.parameters().size() << std::endl;
  std::cout << "cuda Free Memory after Module Copy: " << GetFreeDeviceMemory(0)
            << std::endl;

  // // Check all tensors are on GPU
  // for (const auto& param : module.parameters()) {
  //   std::cout << "param: " << param.device() << std::endl;
  // }s
  // for (const auto& buffer : module.buffers()) {
  //   std::cout << "buffer: " << buffer.device() << std::endl;
  // }

  inputs.clear();
  inputs.push_back(torch::ones({1, 3, 224, 224}, DEFAULT_CUDA_DEVICE));
  std::cout << "cuda Free Memory after Input Create: " << GetFreeDeviceMemory(0)
            << std::endl;

  module.eval();
  output = module.forward(inputs).toTensor();
  // std::cout << "output gpu: " << output << std::endl;
  std::cout << "cuda Free Memory after GPU forward: " << GetFreeDeviceMemory(0)
            << std::endl;
  c10::cuda::CUDACachingAllocator::emptyCache();
  std::cout << "cuda Free Memory after Empty Cache: " << GetFreeDeviceMemory(0)
            << std::endl;

  inputs.clear();
  inputs.push_back(torch::ones({1, 3, 224, 224}, CPU_DEVICE));
  c10::cuda::CUDACachingAllocator::emptyCache();
  std::cout << "cuda Free Memory after Inputs Clear: " << GetFreeDeviceMemory(0)
            << std::endl;

  SetModulePinnedMemory(&module, host_ptr);
  device_memory_pool->FreeMemory(
      id, device_ptr, param_byte_size + buffer_byte_size, DEFAULT_CUDA_DEVICE);
  c10::cuda::CUDACachingAllocator::emptyCache();
  output = module.forward(inputs).toTensor();
  std::cout << "cuda Free Memory after CPU forward: " << GetFreeDeviceMemory(0)
            << std::endl;
  // std::cout << "output cpu: " << output << std::endl;

  device_ptr = device_memory_pool->AllocateMemory(
      id, param_byte_size + buffer_byte_size, DEFAULT_CUDA_DEVICE);
  cudaMemcpy(
      device_ptr, host_ptr, param_byte_size + buffer_byte_size,
      cudaMemcpyHostToDevice);
  SetModuleCudaMemory(&module, device_ptr, DEFAULT_CUDA_DEVICE);

  inputs.clear();
  inputs.push_back(torch::ones({1, 3, 224, 224}, DEFAULT_CUDA_DEVICE));
  output = module.forward(inputs).toTensor();
}

int
main()
{
  std::string model_path = "traced_resnet_model.pt";

  auto* host_memory_pool = HostMemoryPool::GetInstance();
  auto* device_memory_pool = DeviceMemoryPool::GetInstance();

  for (int i = 0; i < 10; i++) {
    inference_cycle(model_path, host_memory_pool, device_memory_pool, i);
  }
}