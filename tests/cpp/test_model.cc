#include <torch/script.h>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <thread>

#define NUM_THREADS 32
#define NUM_MODELS 128

typedef torch::jit::script::Module ScriptModule;
#define DEFAULT_CUDA_DEVICE torch::Device(torch::kCUDA, 0)

int
main()
{
  std::filesystem::path model_repo =
      "/mnt/xly/swap-engine/model_repo_switch-large-128";
  std::string model_name = "switch-large-128_encoder_expert_1_";

  std::vector<std::thread> threads;
  std::vector<ScriptModule*> modules(NUM_MODELS);

  for (int j = 0; j < NUM_THREADS; j++) {
    threads.push_back(std::thread([model_repo, model_name, j, &modules]() {
      for (int i = NUM_MODELS / NUM_THREADS * j;
           i < NUM_MODELS / NUM_THREADS * (j + 1); i++) {
        // auto start_time = std::chrono::system_clock::now();
        std::string model_path =
            model_repo / (model_name + std::to_string(i)) / "0" / "model.pt";
        ScriptModule *module = new ScriptModule(
            torch::jit::load(model_path, torch::Device(torch::kCPU)));
        modules[i] = module;
      }
    }));
  }
  for (auto& thread : threads) {
    thread.join();
  }
  threads.clear();

  auto start_time = std::chrono::system_clock::now();
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

  auto end_time = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end_time - start_time;
  std::cout << "elapsed time: " << elapsed_seconds.count() << "s" << std::endl;
}