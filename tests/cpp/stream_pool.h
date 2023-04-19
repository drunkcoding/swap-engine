#pragma once
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <mutex>
#include <rmm/cuda_stream_pool.hpp>

typedef std::unique_ptr<rmm::cuda_stream_pool> CudaStreamPoolPtr;

class CudaStreamPool {
 public:
  CudaStreamPoolPtr& operator()(const int device_id)
  {
    return cuda_streams_[device_id];
  }

  static CudaStreamPool* GetInstance() { return new CudaStreamPool(); }

 private:
  CudaStreamPool()
  {
    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);
    cuda_streams_.resize(num_devices);
    for (int i = 0; i < num_devices; ++i) {
      cudaSetDevice(i);
      cuda_streams_[i] = std::make_unique<rmm::cuda_stream_pool>(3);
    }
  }
  virtual ~CudaStreamPool() = default;

 private:
  std::vector<CudaStreamPoolPtr> cuda_streams_;
};


extern CudaStreamPool* kCudaStreamPool;
#define CUDA_STREAM_VIEW(device_id, stream_id) \
  (*kCudaStreamPool)(device_id)->get_stream(stream_id)
#define CUDA_STREAM_H2D_VIEW(device_id) CUDA_STREAM_VIEW(device_id, 0)
#define CUDA_STREAM_D2H_VIEW(device_id) CUDA_STREAM_VIEW(device_id, 1)
#define CUDA_STREAM_COMPUTE_VIEW(device_id) CUDA_STREAM_VIEW(device_id, 2)
#define CUDA_STREAM(device_id, stream_id) \
  CUDA_STREAM_VIEW(device_id, stream_id).value()
#define CUDA_STREAM_H2D(device_id) CUDA_STREAM(device_id, 0)
#define CUDA_STREAM_D2H(device_id) CUDA_STREAM(device_id, 1)
#define CUDA_STREAM_COMPUTE(device_id) CUDA_STREAM(device_id, 2)