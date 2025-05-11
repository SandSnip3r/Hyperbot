#ifndef RL_JAX_INTERFACE_HPP_
#define RL_JAX_INTERFACE_HPP_

#include "rl/observation.hpp"

#include <tracy/Tracy.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <atomic>
#include <condition_variable>
#include <optional>
#include <mutex>
#include <vector>

namespace rl {

struct ModelInput {
  // Observations & actions are stacked so that the model can see history.
  // Older observations are at the front and newer observations are at the back.
  std::vector<const Observation*> pastObservationStack;
  std::vector<int> pastActionStack;
  const Observation *currentObservation;
};

namespace detail {
struct ModelInputNumpy {
  pybind11::array_t<float> pastObservationStack;
  pybind11::array_t<float> pastObservationTimestamps;
  pybind11::array_t<float> pastActions;
  pybind11::array_t<float> pastMask;
  pybind11::array_t<float> currentObservation;
};

template <typename T>
pybind11::array_t<T> vector1dToNumpy(const std::vector<T> &vector) {
  pybind11::array_t<T> numpyArray(vector.size());
  std::copy(vector.begin(), vector.end(), numpyArray.mutable_data(0));
  return numpyArray;
}
} // namespace detail

class JaxInterface {
public:
  JaxInterface(int observationStackSize, float gamma, float learningRate) : observationStackSize_(observationStackSize), gamma_(gamma), learningRate_(learningRate) {}
  ~JaxInterface();
  void initialize();

  // `canSendPacket` is used for action masking to limit the rate at which packets are sent.
  int selectAction(const ModelInput &modelInput, bool canSendPacket);

  struct TrainAuxOutput {
    std::vector<float> tdErrors;
    // The following values are means over the batch.
    float meanMinQValue;
    float meanMeanQValue;
    float meanMaxQValue;
  };

  // All vectors should have the same size, this is the batch size.
  TrainAuxOutput train(const std::vector<ModelInput> &pastModelInputs,
                       const std::vector<int> &actionsTaken,
                       const std::vector<bool> &isTerminals,
                       const std::vector<float> &rewards,
                       const std::vector<ModelInput> &currentModelInputs,
                       const std::vector<float> &importanceSamplingWeights);
  void updateTargetModel();
  void updateTargetModelPolyak(float tau);
  void printModels();

  // Note: Parent directory does not need to exist.
  void saveCheckpoint(const std::string &modelCheckpointPath, const std::string &targetModelCheckpointPath, const std::string &optimizerStateCheckpointPath);
  void loadCheckpoint(const std::string &modelCheckpointPath, const std::string &targetModelCheckpointPath, const std::string &optimizerStateCheckpointPath);

  void addScalar(std::string_view name, double yValue, double xValue);
private:
  static constexpr size_t kActionSpaceSize{35}; // TODO: If changed, also change rl::ActionBuilder
  static constexpr int kSeed{0};

  // Store gamma and learning rate as member variables
  const size_t observationStackSize_;
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

  // stackSize is the target size of the observation stack. The given vector might have fewer observations, so the implementation should pad.
  pybind11::object observationStackToNumpy(int stackSize, const std::vector<Observation> &observationStack);

  pybind11::object observationsToNumpy(const std::vector<Observation> &observations);

  // stackSize is the target size of the observation stacks. The given vectors might have fewer observations, so the implementation should pad.
  pybind11::object observationStacksToNumpy(int stackSize, const std::vector<std::vector<Observation>> &observationStacks);

  // Convert a ModelInput to corresponding numpy arrays
  detail::ModelInputNumpy modelInputToNumpy(const ModelInput &modelInput);

  // Convert a vector of ModelInputs to batches of corresponding numpy arrays
  detail::ModelInputNumpy modelInputsToNumpy(const std::vector<ModelInput> &modelInputs);

  size_t writeModelInputToRawArray(const ModelInput &modelInput, float *array);

  // Note, this also works with a default constructed observation.
  size_t getObservationNumpySize(const Observation &observation) const;

  size_t writeEmptyObservationToRawArray(size_t observationSize, float *array);

  // Returns the number of floats written to the array.
  size_t writeObservationToRawArray(const Observation &observation, float *array);

  size_t writeOneHotEvent(event::EventCode eventCode, float *array);

  void writeEmptyActionToRawArray(float *array);

  void writeActionToRawArray(int action, float *array);

  pybind11::array_t<float> createActionMaskNumpy(bool canSendPacket);
};

// We need 3 main interfaces:
// 1. Get an action choice from the latest model.
// 2. Get an action choice from older static model.
// 3. Train the latest model.

} // namespace rl

#endif // RL_JAX_INTERFACE_HPP_