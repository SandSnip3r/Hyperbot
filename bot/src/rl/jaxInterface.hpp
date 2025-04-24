#ifndef RL_JAX_INTERFACE_HPP_
#define RL_JAX_INTERFACE_HPP_

#include "rl/observation.hpp"

#include <tracy/Tracy.hpp>

#include <pybind11/pybind11.h>

#include <atomic>
#include <condition_variable>
#include <optional>
#include <mutex>

namespace rl {

class JaxInterface {
public:
  JaxInterface() = default;
  ~JaxInterface();
  void initialize();
  int selectAction(const Observation &observation, bool canSendPacket);

  struct TrainAuxOutput {
    float tdError;
    float minQValue;
    float meanQValue;
    float maxQValue;
  };
  TrainAuxOutput train(const std::vector<Observation> &olderObservation,
                       const std::vector<int> &actionIndex,
                       const std::vector<bool> &isTerminal,
                       const std::vector<float> &reward,
                       const std::vector<Observation> &newerObservation,
                       const std::vector<float> &weight);
  void updateTargetModel();
  void printModels();

  // Note: Parent directory does not need to exist.
  void saveCheckpoint(const std::string &modelCheckpointPath, const std::string &targetModelCheckpointPath, const std::string &optimizerStateCheckpointPath);
  void loadCheckpoint(const std::string &modelCheckpointPath, const std::string &targetModelCheckpointPath, const std::string &optimizerStateCheckpointPath);

  void addScalar(std::string_view name, double yValue, double xValue);
private:
  static constexpr int kActionSpaceSize{36}; // TODO: If changed, also change rl::ActionBuilder
  static constexpr float kLearningRate{3e-6};
  static constexpr int kSeed{0};
  std::optional<pybind11::module> dqnModule_;
  std::optional<pybind11::module> randomModule_;
  std::optional<pybind11::module> nnxModule_;
  std::optional<pybind11::module> optaxModule_;
  std::optional<pybind11::object> summaryWriter_;
  std::optional<pybind11::object> rngKey_;
  std::optional<pybind11::object> nnxRngs_;
  std::optional<pybind11::object> model_;
  std::optional<pybind11::object> targetModel_;
  std::optional<pybind11::object> optimizerState_;

  // We use a special synchronization routine to give action selection a higher priority than everything else, since it requires low latency.
  // Context: https://github.com/SandSnip3r/ContentionBenchmark
  std::mutex modelMutex_;
  // Tracy lockable does not seem to work with a condition variable.
  // TracyLockableN(std::mutex, modelMutex_, "JaxInterface::modelMutex");
  std::condition_variable modelConditionVariable_;
  std::atomic<bool> waitingToSelectAction_{false};

  pybind11::object getNextRngKey();
  pybind11::object observationToNumpy(const Observation &observation);
  pybind11::object observationsToNumpy(const std::vector<Observation> &observations);
  size_t getObservationNumpySize(const Observation &observation) const;
  void writeObservationToNumpyArray(const Observation &observation, float *array);
  pybind11::object createActionMask(bool canSendPacket);
};

// We need 3 main interfaces:
// 1. Get an action choice from the latest model.
// 2. Get an action choice from older static model.
// 3. Train the latest model.

} // namespace rl

#endif // RL_JAX_INTERFACE_HPP_