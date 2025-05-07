#ifndef RL_JAX_INTERFACE_HPP_
#define RL_JAX_INTERFACE_HPP_

#include "rl/observation.hpp"

#include <tracy/Tracy.hpp>

#include <pybind11/pybind11.h>

#include <atomic>
#include <condition_variable>
#include <optional>
#include <mutex>
#include <vector>

namespace rl {

class JaxInterface {
public:
  JaxInterface(float gamma, float learningRate) : gamma_(gamma), learningRate_(learningRate) {}
  ~JaxInterface();
  void initialize(int observationStackSize);

  // Excpects a stack of observations. Newest is at the back, oldest is at the front.
  int selectAction(int observationStackSize, const std::vector<Observation> &observationStack, bool canSendPacket);

  struct TrainAuxOutput {
    std::vector<float> tdErrors;
    // The following values are means over the batch.
    float meanMinQValue;
    float meanMeanQValue;
    float meanMaxQValue;
  };

  // Excpects stacks of observations. Newest is at the back, oldest is at the front.
  TrainAuxOutput train(int observationStackSize,
                       const std::vector<std::vector<Observation>> &olderObservationStacks,
                       const std::vector<int> &actionIndex,
                       const std::vector<bool> &isTerminal,
                       const std::vector<float> &reward,
                       const std::vector<std::vector<Observation>> &newerObservationStacks,
                       const std::vector<float> &weight);
  void updateTargetModel();
  void updateTargetModelPolyak(float tau);
  void printModels();

  // Note: Parent directory does not need to exist.
  void saveCheckpoint(const std::string &modelCheckpointPath, const std::string &targetModelCheckpointPath, const std::string &optimizerStateCheckpointPath);
  void loadCheckpoint(const std::string &modelCheckpointPath, const std::string &targetModelCheckpointPath, const std::string &optimizerStateCheckpointPath);

  void addScalar(std::string_view name, double yValue, double xValue);
private:
  static constexpr int kActionSpaceSize{35}; // TODO: If changed, also change rl::ActionBuilder
  static constexpr int kSeed{0};

  // Store gamma and learning rate as member variables
  const float gamma_;
  const float learningRate_;

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

  pybind11::object getOptimizer();
  pybind11::object getNextRngKey();
  pybind11::object observationToNumpy(const Observation &observation);
  pybind11::object observationStackToNumpy(int stackSize, const std::vector<Observation> &observationStack);
  pybind11::object observationsToNumpy(const std::vector<Observation> &observations);
  pybind11::object observationStacksToNumpy(int stackSize, const std::vector<std::vector<Observation>> &observationStacks);

  // Note, this also works with a default constructed observation.
  size_t getObservationNumpySize(const Observation &observation) const;

  void writeEmptyObservationToNumpyArray(int observationSize, float *array);

  // Returns the number of floats written to the array.
  size_t writeObservationToRawArray(const Observation &observation, float *array);

  size_t writeOneHotEvent(event::EventCode eventCode, float *array);

  pybind11::object createActionMask(bool canSendPacket);
};

// We need 3 main interfaces:
// 1. Get an action choice from the latest model.
// 2. Get an action choice from older static model.
// 3. Train the latest model.

} // namespace rl

#endif // RL_JAX_INTERFACE_HPP_