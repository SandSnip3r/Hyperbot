#ifndef RL_JAX_INTERFACE_HPP_
#define RL_JAX_INTERFACE_HPP_

#include "event/eventCode.hpp"
#include "rl/modelInputs.hpp"
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
  void initialize(float dropoutRate);

  class Model {
  public:
    ~Model();
  private:
    Model(pybind11::object model);
    std::optional<pybind11::object> model_;
    friend class JaxInterface;
  };

  Model getModel() const;
  Model getTargetModel() const;
  Model getDummyModel() const;

  class Optimizer {
  public:
    ~Optimizer();
  private:
    Optimizer(pybind11::object optimizer);
    std::optional<pybind11::object> optimizer_;
    friend class JaxInterface;
  };

  Optimizer getOptimizer() const;
  Optimizer getDummyOptimizer() const;

  struct ActionSelectionResult {
    int actionIndex{0};
    std::vector<float> qValues;
  };

  // `canSendPacket` is used for action masking to limit the rate at which packets are sent.
  ActionSelectionResult selectAction(const model_inputs::ModelInputView &modelInputView, bool canSendPacket, std::optional<std::string> metadata = std::nullopt);

  struct TrainAuxOutput {
    float globalNorm;
    std::vector<float> tdErrors;
    // The following values are means over the batch.
    float meanMinQValue;
    float meanMeanQValue;
    float meanMaxQValue;
  };

  // All vectors should have the same size, this is the batch size.
  TrainAuxOutput train(const Model &model,
                       const Optimizer &optimizer,
                       const Model &targetModel,
                       const std::vector<model_inputs::ModelInputView> &pastModelInputViews,
                       const std::vector<int> &actionsTaken,
                       const std::vector<bool> &isTerminals,
                       const std::vector<float> &rewards,
                       const std::vector<model_inputs::ModelInputView> &currentModelInputViews,
                       const std::vector<float> &importanceSamplingWeights);
  void updateTargetModel();
  void updateTargetModelPolyak(float tau);
  void printModels();

  // Note: Parent directory does not need to exist.
  void saveCheckpoint(const std::string &modelCheckpointPath, const std::string &targetModelCheckpointPath, const std::string &optimizerStateCheckpointPath);
  void loadCheckpoint(const std::string &modelCheckpointPath, const std::string &targetModelCheckpointPath, const std::string &optimizerStateCheckpointPath);
private:
  static constexpr int kSeed{0};

  // Store gamma and learning rate as member variables
  const size_t observationStackSize_;
  const float gamma_;
  const float learningRate_;

  std::optional<pybind11::module> dqnModule_;
  std::optional<pybind11::module> randomModule_;
  std::optional<pybind11::module> nnxModule_;
  std::optional<pybind11::module> optaxModule_;
  std::optional<pybind11::object> rngKey_;
  std::optional<pybind11::object> nnxRngs_;
  std::optional<pybind11::object> model_;
  std::optional<pybind11::object> targetModel_;
  std::optional<pybind11::object> optimizerState_;

  // We use a special synchronization routine to give action selection a higher priority than everything else, since it requires low latency.
  // Context: https://github.com/SandSnip3r/ContentionBenchmark
  mutable TracyLockableN(std::mutex, modelMutex_, "JaxInterface::modelMutex");
  std::condition_variable_any modelConditionVariable_;
  std::atomic<bool> waitingToSelectAction_{false};

  pybind11::object getPythonOptimizer();
  pybind11::object getNextRngKey();

  // Convert a ModelInput to corresponding numpy arrays
  detail::ModelInputNumpy modelInputToNumpy(const model_inputs::ModelInputView &modelInputView);

  // Convert a vector of ModelInputs to batches of corresponding numpy arrays
  detail::ModelInputNumpy modelInputsToNumpy(const std::vector<model_inputs::ModelInputView> &modelInputViews);

  size_t writeZerosToRawArray(size_t count, float *array);

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