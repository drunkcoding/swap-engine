#include <cuda_runtime_api.h>
#include <sys/types.h>

#include <iostream>
// Compile with g++ alloc.cc -o alloc.so -I/usr/local/cuda/include -shared -fPIC
extern "C" {
void*
my_malloc(ssize_t size, int device, cudaStream_t stream)
{
  void* ptr;
  cudaMallocManaged(&ptr, size);
  //   std::cout << "alloc " << ptr << size << std::endl;
  return ptr;
}

void
my_free(void* ptr, ssize_t size, int device, cudaStream_t stream)
{
  //   std::cout << "free " << ptr << " " << stream << std::endl;
  cudaFree(ptr);
}
}