#include <chrono>
#include <iostream>
#include <cstring>
#include <thread>
#include <vector>
#include <atomic>

#define NUM_THREADS 1

int
main()
{
  std::vector<std::thread> threads;
  std::vector<void*> src(NUM_THREADS);
  std::vector<void*> dst(NUM_THREADS);

  for (int i = 0; i < NUM_THREADS; i++) {
    src[i] = malloc(1024 * 100);
    dst[i] = malloc(1024 * 100);
  }

  std::atomic<uint64_t> start_time{0};
  uint64_t init = 0;

  // auto start_time = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < NUM_THREADS; i++) {
    threads.push_back(std::thread([i, &src, &dst, &start_time] {
      // repeat the memcpy 100 times to make sure the test takes long enough
      uint64_t init = 0;
      auto start = std::chrono::high_resolution_clock::now();
      start_time.compare_exchange_strong(init, start.time_since_epoch().count());
      for (int j = 0; j < 100000; j++) memcpy(dst[i], src[i], 1024);
      // std::chrono::duration<double> diff = end - start;
      // std::cout << "Thread " << i << " took " << diff.count() << " seconds"
                // << std::endl;
    }));
  }
  

  for (int i = 0; i < NUM_THREADS; i++) {
    threads[i].join();
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  std::cout << "Total time: " << end_time.time_since_epoch().count() - start_time << " ns" << std::endl;

  for (int i = 0; i < NUM_THREADS; i++) {
    free(src[i]);
    free(dst[i]);
  }
}