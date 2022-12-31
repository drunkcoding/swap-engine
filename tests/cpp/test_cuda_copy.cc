#include <cuda_runtime.h>

#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

#define NUM_THREADS 64

int
main()
{
  std::vector<void*> src(NUM_THREADS);
  std::vector<void*> dst(NUM_THREADS);
  std::vector<std::thread> threads;

  cudaSetDevice(0);

  void* warmup_src = malloc(1024 * 18);
  void* warmup_dst;

  cudaMalloc(&warmup_dst, 1024 * 18);
  cudaMemcpy(warmup_dst, warmup_src, 1024 * 18, cudaMemcpyHostToDevice);

  for (int i = 0; i < NUM_THREADS; i++) {
    src[i] = malloc(1024 * 9);

    // allocate memory on gpu 0 with cuda Malloc
    
    cudaMalloc(&dst[i], 1024 * 9);
  }

  auto start_time = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < NUM_THREADS; i++) {
    threads.push_back(std::thread([i, &src, &dst] {
      // repeat the memcpy 100 times to make sure the test takes long enough
      for (int j = 0; j < 1000; j++)
        cudaMemcpy(dst[i], src[i], 1024 * 9, cudaMemcpyHostToDevice);
    }));
  }


  for (int i = 0; i < NUM_THREADS; i++) {
    threads[i].join();
  }
  
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end_time - start_time;
  std::cout << "Total time: " << elapsed_seconds.count() << " seconds"
            << std::endl;

  for (int i = 0; i < NUM_THREADS; i++) {
    free(src[i]);
    cudaFree(dst[i]);
  }

  cudaFree(warmup_dst);
  free(warmup_src);
}