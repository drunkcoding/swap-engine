#pragma once
#include <torch/script.h>

#include <mutex>
#include <rmm/mr/device/arena_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/host/pinned_memory_resource.hpp>
#include <unordered_set>
#include <vector>

#include "stream_pool.h"

typedef rmm::mr::cuda_memory_resource CudaMemoryResource;
typedef std::shared_ptr<CudaMemoryResource> CudaMemoryResourcePtr;
typedef rmm::mr::pinned_memory_resource PinnedMemoryResource;
typedef std::shared_ptr<PinnedMemoryResource> PinnedMemoryResourcePtr;
typedef rmm::mr::arena_memory_resource<CudaMemoryResource> ArenaMemoryResource;
typedef std::shared_ptr<ArenaMemoryResource> ArenaMemoryResourcePtr;


class HostMemoryPool {
 public:
  static HostMemoryPool* GetInstance() { return new HostMemoryPool(); }

  void* AllocateMemory(
      const std::size_t key, const std::int64_t size,
      const torch::Device& device)
  {
    assert(device.is_cpu());
    std::unique_lock lock(mutex_);
    if (allocated_id_.find(key) != allocated_id_.end()) {
      return nullptr;
    }
    allocated_id_.insert(key);
    return pinned_mr_.allocate(size);
  }

  int FreeMemory(
      const std::size_t key, void* data, const std::int64_t size,
      const torch::Device& device)
  {
    assert(device.is_cpu());
    std::unique_lock lock(mutex_);
    if (allocated_id_.find(key) == allocated_id_.end()) {
      return -1;
    }
    allocated_id_.erase(key);
    pinned_mr_.deallocate(data, size);
    return 0;
  }

 private:
  HostMemoryPool() = default;
  virtual ~HostMemoryPool() = default;

 private:
  std::unordered_set<std::uint64_t> allocated_id_;
  rmm::mr::pinned_memory_resource pinned_mr_;
  std::mutex mutex_;
};

class DeviceMemoryPool {
 public:
  static DeviceMemoryPool* GetInstance() { return new DeviceMemoryPool(); }

  void* AllocateMemory(
      const std::size_t key, const std::int64_t size,
      const torch::Device& device)
  {
    int device_id = device.index();
    std::unique_lock lock(mutex_);
    std::cout << "device id: " << device_id << " key: " << key << std::endl;
    if (allocated_id_[device_id].find(key) != allocated_id_[device_id].end()) {
      return nullptr;
    }
    allocated_id_[device_id].insert(key);
    return arena_mr_[device_id]->allocate(
        size, CUDA_STREAM_H2D_VIEW(device_id));
  }

  int FreeMemory(
      const std::size_t key, void* data, const std::int64_t size,
      const torch::Device& device)
  {
    int device_id = device.index();
    std::unique_lock lock(mutex_);
    if (allocated_id_[device_id].find(key) == allocated_id_[device_id].end()) {
      return -1;
    }
    allocated_id_[device_id].erase(key);
    arena_mr_[device_id]->deallocate(
        data, size, CUDA_STREAM_H2D_VIEW(device_id));
    return 0;
  }

 private:
  DeviceMemoryPool()
  {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    std::cout << "device count: " << device_count << std::endl;

    for (int i = 0; i < device_count; ++i) {
      cudaSetDevice(i);
      cuda_mr_.emplace_back(std::make_shared<CudaMemoryResource>());
      arena_mr_.emplace_back(std::make_shared<ArenaMemoryResource>(
          cuda_mr_[i].get(), 20 * 1024 * 1024 * 1024ULL, false));
      std::unordered_set<std::uint64_t> allocated_id;
      allocated_id_.emplace_back(allocated_id);
    }
  }
  virtual ~DeviceMemoryPool() = default;

 private:
  std::vector<std::unordered_set<std::uint64_t>> allocated_id_;
  std::vector<CudaMemoryResourcePtr> cuda_mr_;
  std::vector<ArenaMemoryResourcePtr> arena_mr_;
  std::mutex mutex_;
};
