#include <torch/script.h>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <thread>

#define NUM_THREADS 4
#define NUM_MODELS 64

typedef torch::jit::script::Module ScriptModule;
#define DEFAULT_CUDA_DEVICE torch::Device(torch::kCUDA, 0)
#define CPU_DEVICE torch::Device(torch::kCPU)

int
main()
{
  std::filesystem::path model_repo =
      "/mnt/raid0nvme1/xly/swap-engine/model_repo_switch-base-128";
  std::string model_name = "switch-base-128_decoder_expert_1_";
  // /mnt/raid0nvme1/xly/swap-engine/model_repo_switch-base-128/switch-base-128_decoder_expert_1_15
  std::vector<std::thread> threads;
  std::vector<ScriptModule*> modules(NUM_MODELS);

  auto start_time = std::chrono::system_clock::now();
  for (int j = 0; j < NUM_THREADS; j++) {
    threads.push_back(std::thread([model_repo, model_name, j, &modules]() {
      for (int i = NUM_MODELS / NUM_THREADS * j;
           i < NUM_MODELS / NUM_THREADS * (j + 1); i++) {
        // auto start_time = std::chrono::system_clock::now();
        std::string model_path =
            model_repo / (model_name + std::to_string(i)) / "0" / "model.pt";
        ScriptModule* module = new ScriptModule(
            torch::jit::load(model_path, torch::Device(torch::kCUDA, 0)));
        modules[i] = module;
      }
    }));
  }
  for (auto& thread : threads) {
    thread.join();
  }
  threads.clear();

  auto end_time = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end_time - start_time;
  std::cout << "elapsed time: " << elapsed_seconds.count() << "s" << std::endl;

  start_time = std::chrono::system_clock::now();
  for (int j = 0; j < NUM_THREADS; j++) {
    threads.push_back(std::thread([model_repo, model_name, j, &modules]() {
      for (int i = NUM_MODELS / NUM_THREADS * j;
           i < NUM_MODELS / NUM_THREADS * (j + 1); i++) {
        auto start = std::chrono::system_clock::now();
        std::string model_path =
            model_repo / (model_name + std::to_string(i)) / "0" / "model.pt";
        ScriptModule* module = new ScriptModule(
            torch::jit::load(model_path, torch::Device(torch::kCPU)));
        modules[i] = module;
        auto end = std::chrono::system_clock::now();
        // std::chrono::duration<double> elapsed_seconds = end - start;
        // std::cout << "thread time: " << elapsed_seconds.count() << "s"
        //           << " start time: " << start.time_since_epoch().count()
        //           << " end time: " << end.time_since_epoch().count()
        //           << std::endl;
      }
    }));
  }
  for (auto& thread : threads) {
    thread.join();
  }
  threads.clear();

  end_time = std::chrono::system_clock::now();
  elapsed_seconds = end_time - start_time;
  std::cout << "elapsed time: " << elapsed_seconds.count() << "s" << std::endl;

  for (int j = 0; j < NUM_THREADS; j++) {
    threads.push_back(std::thread([model_repo, model_name, j, &modules]() {
      for (int i = NUM_MODELS / NUM_THREADS * j;
           i < NUM_MODELS / NUM_THREADS * (j + 1); i++) {
        modules[i]->to(DEFAULT_CUDA_DEVICE);
      }
    }));
  }
  for (auto& thread : threads) {
    thread.join();
  }
  threads.clear();

  for (int j = 0; j < NUM_THREADS; j++) {
    threads.push_back(std::thread([model_repo, model_name, j, &modules]() {
      for (int i = NUM_MODELS / NUM_THREADS * j;
           i < NUM_MODELS / NUM_THREADS * (j + 1); i++) {
        modules[i]->to(CPU_DEVICE);
      }
    }));
  }
  for (auto& thread : threads) {
    thread.join();
  }
  threads.clear();

  start_time = std::chrono::system_clock::now();
  for (int j = 0; j < NUM_THREADS; j++) {
    threads.push_back(std::thread([model_repo, model_name, j, &modules]() {
      auto start = std::chrono::system_clock::now();
      for (int i = NUM_MODELS / NUM_THREADS * j;
           i < NUM_MODELS / NUM_THREADS * (j + 1); i++) {
        modules[i]->to(DEFAULT_CUDA_DEVICE);
      }
      auto end = std::chrono::system_clock::now();
      // std::chrono::duration<double> elapsed_seconds = end - start;
      // std::cout << "thread elapsed time: " << elapsed_seconds.count() << "s"
      //           << " start time: " << start.time_since_epoch().count()
      //           << " end time: " << end.time_since_epoch().count() << std::endl;
    }));
  }
  for (auto& thread : threads) {
    thread.join();
  }
  threads.clear();

  end_time = std::chrono::system_clock::now();
  elapsed_seconds = end_time - start_time;
  std::cout << "elapsed time: " << elapsed_seconds.count() << "s" << std::endl;
}