#ifndef RL_JAX_INTERFACE_HPP_
#define RL_JAX_INTERFACE_HPP_

#include "rl/observation.hpp"

#include <tracy/Tracy.hpp>

#include <pybind11/pybind11.h>

#include <optional>
#include <mutex>

namespace rl {

class JaxInterface {
public:
  JaxInterface() = default;
  ~JaxInterface();
  void initialize();
  int selectAction(const Observation &observation, bool canSendPacket);
  void train(const Observation &olderObservation, int actionIndex, bool isTerminal, float reward, const Observation &newerObservation);
  void updateTargetModel();
  void printModels();
private:
  static constexpr int kActionSpaceSize{36}; // TODO: If changed, also change rl::ActionBuilder
  static constexpr float kLearningRate{1e-5};
  static constexpr int kSeed{0};
  std::optional<pybind11::module> dqnModule_;
  std::optional<pybind11::module> randomModule_;
  std::optional<pybind11::module> nnxModule_;
  std::optional<pybind11::object> rngKey_;
  std::optional<pybind11::object> nnxRngs_;
  std::optional<pybind11::object> model_;
  std::optional<pybind11::object> targetModel_;
  std::optional<pybind11::object> optimizerState_;
  TracyLockableN(std::mutex, modelMutex_, "JaxInterface::modelMutex");
  TracyLockableN(std::mutex, targetModelMutex_, "JaxInterface::targetModelMutex");

  pybind11::object getNextRngKey();
  pybind11::object observationToNumpy(const Observation &observation);
  pybind11::object createActionMask(bool canSendPacket);
};

// We need 3 main interfaces:
// 1. Get an action choice from the latest model.
// 2. Get an action choice from older static model.
// 3. Train the latest model.

} // namespace rl

#endif // RL_JAX_INTERFACE_HPP_