#ifndef CONSTS_H_
#define CONSTS_H_

namespace swap_engine {
enum class TensorState {
  kGPURunning = 0,
  kGPUIdle = 1,
  kCPUIdle = 2,
  kNVMeIdle = 3
};
}  // namespace swap_engine

#endif /* CONSTS_H_ */