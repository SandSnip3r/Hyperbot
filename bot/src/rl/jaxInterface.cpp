#include "flags.hpp"
#include "rl/jaxInterface.hpp"
#include "rl/actionSpace.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <tracy/Tracy.hpp>

#include <absl/flags/flag.h>
#include <absl/log/log.h>
#include <absl/strings/str_join.h>
#include <absl/strings/str_format.h>

#include <algorithm>
#include <fstream>
#include <thread>

namespace py = pybind11;

namespace rl {

namespace {

bool checkHasAttr(py::object &obj, const std::string &attr_name) {
  // Get the hasattr function from Python's built-in module
  py::object hasattr_func = py::module_::import("builtins").attr("hasattr");

  // Call hasattr function
  return hasattr_func(obj, attr_name).cast<bool>();
}

void logToCsvFile(const std::string &metadata, const std::chrono::steady_clock::time_point &timestamp, const detail::ModelInputNumpy &modelInputNumpy, const std::vector<float> &qValues, int actionIndex) {
  // TODO: For now, we've disabled this.
  return;
  // Log the following to a CSV file:
  // ms_since_epoch, pvp_id, *observation_data, action_index, *q_values
  std::chrono::milliseconds msSinceEpoch = std::chrono::duration_cast<std::chrono::milliseconds>(timestamp.time_since_epoch());
  static bool firstRun = true;
  if (firstRun) {
    // Write the header only once.
    constexpr std::array headers = {
      "ms_since_epoch",
      "pvp_id",
      "po0_hp_pot_avail",
      "po0_hp_pot_count",
      "po0_hp",
      "po0_opp_hp",
      "po1_hp_pot_avail",
      "po1_hp_pot_count",
      "po1_hp",
      "po1_opp_hp",
      "po2_hp_pot_avail",
      "po2_hp_pot_count",
      "po2_hp",
      "po2_opp_hp",
      "po3_hp_pot_avail",
      "po3_hp_pot_count",
      "po3_hp",
      "po3_opp_hp",
      "po4_hp_pot_avail",
      "po4_hp_pot_count",
      "po4_hp",
      "po4_opp_hp",
      "po5_hp_pot_avail",
      "po5_hp_pot_count",
      "po5_hp",
      "po5_opp_hp",
      "po6_hp_pot_avail",
      "po6_hp_pot_count",
      "po6_hp",
      "po6_opp_hp",
      "po7_hp_pot_avail",
      "po7_hp_pot_count",
      "po7_hp",
      "po7_opp_hp",
      "po0_timestamp",
      "po1_timestamp",
      "po2_timestamp",
      "po3_timestamp",
      "po4_timestamp",
      "po5_timestamp",
      "po6_timestamp",
      "po7_timestamp",
      "po0_action_sleep",
      "po0_action_common_attack",
      "po0_action_hp_pot",
      "po1_action_sleep",
      "po1_action_common_attack",
      "po1_action_hp_pot",
      "po2_action_sleep",
      "po2_action_common_attack",
      "po2_action_hp_pot",
      "po3_action_sleep",
      "po3_action_common_attack",
      "po3_action_hp_pot",
      "po4_action_sleep",
      "po4_action_common_attack",
      "po4_action_hp_pot",
      "po5_action_sleep",
      "po5_action_common_attack",
      "po5_action_hp_pot",
      "po6_action_sleep",
      "po6_action_common_attack",
      "po6_action_hp_pot",
      "po7_action_sleep",
      "po7_action_common_attack",
      "po7_action_hp_pot",
      "po0_mask",
      "po1_mask",
      "po2_mask",
      "po3_mask",
      "po4_mask",
      "po5_mask",
      "po6_mask",
      "po7_mask",
      "curr_hp_pot_avail",
      "curr_hp_pot_count",
      "curr_hp",
      "curr_opp_hp",
      "q_val_sleep",
      "q_val_common_attack",
      "q_val_hp_pot",
      "action_index"
    };
    std::ofstream csvFile("intelligence_actor_log.csv");
    csvFile << absl::StrJoin(headers, ",") << std::endl;
    firstRun = false;
  }
  std::ofstream csvFile("intelligence_actor_log.csv", std::ios::app);
  csvFile << msSinceEpoch.count() << ','
          << metadata << ',';
  auto printFlattenedArray = [&csvFile](const py::array_t<float> &array) {
    // Request a buffer info object
    py::buffer_info buffer = array.request();

    // Get the pointer to the data
    float* data_ptr = static_cast<float*>(buffer.ptr);

    // Iterate over the data as a flattened array
    for (size_t i=0; i<buffer.size; ++i) {
      csvFile << data_ptr[i] << ',';
    }
  };
  printFlattenedArray(modelInputNumpy.pastObservationStack);
  printFlattenedArray(modelInputNumpy.pastObservationTimestamps);
  printFlattenedArray(modelInputNumpy.pastActions);
  printFlattenedArray(modelInputNumpy.pastMask);
  printFlattenedArray(modelInputNumpy.currentObservation);
  for (const float f : qValues) {
    csvFile << f << ',';
  }
  csvFile << actionIndex << '\n';
}

} // namespace

JaxInterface::~JaxInterface() {
  VLOG(1) << "Destructing JaxInterface";
  // Manually destroy all python objects by first acquiring the GIL.
  py::gil_scoped_acquire acquire;
  if (dqnModule_.has_value()) {
    dqnModule_.reset();
  }
  if (randomModule_.has_value()) {
    randomModule_.reset();
  }
  if (nnxModule_.has_value()) {
    nnxModule_.reset();
  }
  if (optaxModule_.has_value()) {
    optaxModule_.reset();
  }
  if (rngKey_.has_value()) {
    rngKey_.reset();
  }
  if (nnxRngs_.has_value()) {
    nnxRngs_.reset();
  }
  if (model_.has_value()) {
    model_.reset();
  }
  if (targetModel_.has_value()) {
    targetModel_.reset();
  }
  if (optimizerState_.has_value()) {
    optimizerState_.reset();
  }
}

void JaxInterface::initialize(float dropoutRate) {
  using namespace pybind11::literals;

  VLOG(1) << "Constructing JaxInterface";
  {
    std::unique_lock modelLock(modelMutex_);
    modelConditionVariable_.wait(modelLock, [this]() -> bool {
      return !waitingToSelectAction_.load();
    });
    py::gil_scoped_acquire acquire;
    py::object DqnModelType;
    py::tuple graphAndWeights;

    // Load our python module which has our RL code.
    dqnModule_ = py::module::import("rl.python.dqn");

    // Grab a random key based on our seed. Any randomness from this point on will split & replace this key held in member data.
    randomModule_ = py::module::import("jax.random");
    rngKey_ = randomModule_->attr("key")(kSeed);
    // NNX's Rngs is created using a JAX key, so we'll use the above key to create our NNX Rngs.
    nnxModule_ = py::module::import("flax.nnx");
    nnxRngs_ = nnxModule_->attr("Rngs")(getNextRngKey());
    // Now, we want to create a randomly initialized model. Specifically, we want randomly initialized weights. To do this, we'll instantiate our NNX model, then split the abstract graph and the concrete weights.
    DqnModelType = dqnModule_->attr("DqnModel");

    constexpr int kObservationSize = Observation::size();
    constexpr int kActionSpaceSize = ActionSpace::size();
    model_ = DqnModelType(kObservationSize, observationStackSize_, kActionSpaceSize, dropoutRate, *nnxRngs_);
    targetModel_ = py::module::import("copy").attr("deepcopy")(*model_);

    optaxModule_ = py::module::import("optax");
    py::object optimizer = getPythonOptimizer();
    optimizerState_ = nnxModule_->attr("Optimizer")(*model_, optimizer);
  }
  modelConditionVariable_.notify_all();
}

JaxInterface::Model::Model(pybind11::object model) : model_(model) {}
JaxInterface::Model::~Model() {
  if (model_) {
    py::gil_scoped_acquire acquire;
    model_.reset();
  }
}

JaxInterface::Model JaxInterface::getModel() const {
  if (!model_) {
    throw std::runtime_error("model is not yet initialized");
  }
  std::unique_lock modelLock(modelMutex_);
  py::gil_scoped_acquire acquire;
  return Model(*model_);
}

JaxInterface::Model JaxInterface::getTargetModel() const {
  if (!targetModel_) {
    throw std::runtime_error("targetModel is not yet initialized");
  }
  std::unique_lock modelLock(modelMutex_);
  py::gil_scoped_acquire acquire;
  return Model(*targetModel_);
}

JaxInterface::Model JaxInterface::getDummyModel() const {
  if (!model_) {
    throw std::runtime_error("model is not yet initialized");
  }
  std::unique_lock modelLock(modelMutex_);
  py::gil_scoped_acquire acquire;
  return Model(py::module::import("copy").attr("deepcopy")(*model_));
}

JaxInterface::Optimizer::Optimizer(pybind11::object optimizer) : optimizer_(optimizer) {}
JaxInterface::Optimizer::~Optimizer() {
  if (optimizer_) {
    py::gil_scoped_acquire acquire;
    optimizer_.reset();
  }
}

JaxInterface::Optimizer JaxInterface::getOptimizer() const {
  if (!optimizerState_) {
    throw std::runtime_error("optimizerState is not yet initialized");
  }
  std::unique_lock modelLock(modelMutex_);
  py::gil_scoped_acquire acquire;
  return Optimizer(*optimizerState_);
}

JaxInterface::Optimizer JaxInterface::getDummyOptimizer() const {
  if (!optimizerState_) {
    throw std::runtime_error("optimizerState is not yet initialized");
  }
  std::unique_lock modelLock(modelMutex_);
  py::gil_scoped_acquire acquire;
  return Optimizer(py::module::import("copy").attr("deepcopy")(*optimizerState_));
}

JaxInterface::ActionSelectionResult JaxInterface::selectAction(const model_inputs::ModelInputView &modelInputView, bool canSendPacket, std::optional<std::string> metadata) {
  ZoneScopedN("JaxInterface::selectAction");
  ActionSelectionResult result;
  try {
    waitingToSelectAction_ = true;
    std::unique_lock modelLock(modelMutex_);
    waitingToSelectAction_ = false;
    py::gil_scoped_acquire acquire;
    ZoneScopedN("JaxInterface::selectAction_GIL");
    // Convert C++ observation into numpy observation
    detail::ModelInputNumpy numpyModelInput = modelInputToNumpy(modelInputView);
    // Create an action mask based on whether or not we can send a packet
    py::array_t<float> actionMask = createActionMaskNumpy(canSendPacket);
    // Get the action from the model
    py::object actionPyObject;
    {
      ZoneScopedN("JaxInterface::selectAction_PYTHON");
      // Input is:
      // - pastObservationStack:      (stackSize, observationSize)
      // - pastObservationTimestamps: (stackSize, 1)
      // - pastActions:               (stackSize, actionSpaceSize)
      // - pastMask:                  (stackSize, 1)
      // - currentObservation:        (observationSize)
      // - actionMask:                (actionSpaceSize)
      actionPyObject = dqnModule_->attr("selectAction")(*model_,
                                                        numpyModelInput.pastObservationStack,
                                                        numpyModelInput.pastObservationTimestamps,
                                                        numpyModelInput.pastActions,
                                                        numpyModelInput.pastMask,
                                                        numpyModelInput.currentObservation,
                                                        actionMask);
    }
    py::tuple resultTuple = actionPyObject.cast<py::tuple>();
    result.actionIndex = resultTuple[0].cast<int>();
    py::array_t<float> qValuesArray = resultTuple[1].cast<py::array_t<float>>();
    result.qValues.assign(qValuesArray.data(), qValuesArray.data() + qValuesArray.size());
    if (metadata) {
      logToCsvFile(metadata.value(), modelInputView.currentObservation->timestamp_, numpyModelInput, result.qValues, result.actionIndex);
    }
  } catch (std::exception &ex) {
    LOG(ERROR) << "Caught exception in JaxInterface::selectAction: " << ex.what();
    modelConditionVariable_.notify_all();
    throw;
  }
  modelConditionVariable_.notify_all();
  VLOG(1) << "Chose action " << result.actionIndex;
  return result;
}

JaxInterface::TrainAuxOutput JaxInterface::train(const Model &model,
                                                 const Optimizer &optimizer,
                                                 const Model &targetModel,
                                                 const std::vector<model_inputs::ModelInputView> &pastModelInputViews,
                                                 const std::vector<int> &actionsTaken,
                                                 const std::vector<bool> &isTerminals,
                                                 const std::vector<float> &rewards,
                                                 const std::vector<model_inputs::ModelInputView> &currentModelInputViews,
                                                 const std::vector<float> &importanceSamplingWeights) {
  ZoneScopedN("JaxInterface::train");
  {
    size_t size = pastModelInputViews.size();
    if (size == 0 ||
        actionsTaken.size() != size ||
        isTerminals.size() != size ||
        rewards.size() != size ||
        currentModelInputViews.size() != size ||
        importanceSamplingWeights.size() != size) {
      throw std::runtime_error(absl::StrFormat("JaxInterface::train: Batch size mismatch: %zu %zu %zu %zu %zu %zu",
        pastModelInputViews.size(), actionsTaken.size(), isTerminals.size(), rewards.size(), currentModelInputViews.size(), importanceSamplingWeights.size()));
    }
  }
  TrainAuxOutput result;
  {
    std::unique_lock modelLock(modelMutex_);
    modelConditionVariable_.wait(modelLock, [this]() -> bool {
      return !waitingToSelectAction_.load();
    });
    py::gil_scoped_acquire acquire;
    ZoneScopedN("JaxInterface::train_GIL");
    try {
      detail::ModelInputNumpy pastModelInputsNumpy = modelInputsToNumpy(pastModelInputViews);
      py::tuple pastModelInputTuple = py::make_tuple(
        pastModelInputsNumpy.pastObservationStack,
        pastModelInputsNumpy.pastObservationTimestamps,
        pastModelInputsNumpy.pastActions,
        pastModelInputsNumpy.pastMask,
        pastModelInputsNumpy.currentObservation
      );
      detail::ModelInputNumpy currentModelInputsNumpy = modelInputsToNumpy(currentModelInputViews);
      py::tuple currentModelInputTuple = py::make_tuple(
        currentModelInputsNumpy.pastObservationStack,
        currentModelInputsNumpy.pastObservationTimestamps,
        currentModelInputsNumpy.pastActions,
        currentModelInputsNumpy.pastMask,
        currentModelInputsNumpy.currentObservation
      );

      py::object auxOutput;
      {
        ZoneScopedN("JaxInterface::train_PYTHON");
        auxOutput = dqnModule_->attr("jittedTrain")(model.model_,
                                                    optimizer.optimizer_,
                                                    targetModel.model_,
                                                    pastModelInputTuple,
                                                    detail::vector1dToNumpy(actionsTaken),
                                                    detail::vector1dToNumpy(isTerminals),
                                                    detail::vector1dToNumpy(rewards),
                                                    currentModelInputTuple,
                                                    detail::vector1dToNumpy(importanceSamplingWeights),
                                                    gamma_,
                                                    getNextRngKey());
      }
      py::tuple auxOutputTuple = auxOutput.cast<py::tuple>();
      result.globalNorm = auxOutputTuple[0].cast<float>();
      result.tdErrors = auxOutputTuple[1].cast<std::vector<float>>();
      result.meanMinQValue = auxOutputTuple[2].cast<float>();
      result.meanMeanQValue = auxOutputTuple[3].cast<float>();
      result.meanMaxQValue = auxOutputTuple[4].cast<float>();
    } catch (std::exception &ex) {
      LOG(ERROR) << "Caught exception in JaxInterface::train: " << ex.what();
      throw;
    }
  }
  modelConditionVariable_.notify_all();
  return result;
}

void JaxInterface::updateTargetModel() {
  ZoneScopedN("JaxInterface::updateTargetModel");
  {
    std::unique_lock modelLock(modelMutex_);
    modelConditionVariable_.wait(modelLock, [this]() -> bool {
      return !waitingToSelectAction_.load();
    });
    py::gil_scoped_acquire acquire;
    try {
      targetModel_ = dqnModule_->attr("getCopyOfModel")(*model_, *targetModel_);
    } catch (std::exception &ex) {
      LOG(ERROR) << "Caught exception in JaxInterface::updateTargetModel: " << ex.what();
      throw;
    }
  }
  modelConditionVariable_.notify_all();
}

void JaxInterface::updateTargetModelPolyak(float tau) {
  ZoneScopedN("JaxInterface::updateTargetModelPolyak");
  {
    std::unique_lock modelLock(modelMutex_);
    modelConditionVariable_.wait(modelLock, [this]() -> bool {
      return !waitingToSelectAction_.load();
    });
    py::gil_scoped_acquire acquire;
    try {
      targetModel_ = dqnModule_->attr("polyakUpdateTargetModel")(*model_, *targetModel_, tau);
    } catch (std::exception &ex) {
      LOG(ERROR) << "Caught exception in JaxInterface::updateTargetModelPolyak: " << ex.what();
      throw;
    }
  }
  modelConditionVariable_.notify_all();
}

void JaxInterface::printModels() {
  try {
    {
      std::unique_lock modelLock(modelMutex_);
      modelConditionVariable_.wait(modelLock, [this]() -> bool {
        return !waitingToSelectAction_.load();
      });
      py::gil_scoped_acquire acquire;
      dqnModule_->attr("printWeights")(*model_);
      dqnModule_->attr("printWeights")(*targetModel_);
    }
    modelConditionVariable_.notify_all();
  } catch (std::exception &ex) {
    LOG(ERROR) << "Caught exception in JaxInterface::printModels: " << ex.what();
  }
}

void JaxInterface::saveCheckpoint(const std::string &modelCheckpointPath, const std::string &targetModelCheckpointPath, const std::string &optimizerStateCheckpointPath) {
  LOG(INFO) << "Saving checkpoint at paths \"" << modelCheckpointPath << "\", \"" << targetModelCheckpointPath << "\", and \"" << optimizerStateCheckpointPath << "\"";
  try {
    std::unique_lock modelLock(modelMutex_);
    modelConditionVariable_.wait(modelLock, [this]() -> bool {
      return !waitingToSelectAction_.load();
    });
    py::gil_scoped_acquire acquire;
    dqnModule_->attr("checkpointModel")(*model_, modelCheckpointPath);
    dqnModule_->attr("checkpointModel")(*targetModel_, targetModelCheckpointPath);
    dqnModule_->attr("checkpointOptimizer")(*optimizerState_, optimizerStateCheckpointPath);
  } catch (std::exception &ex) {
    LOG(ERROR) << "Caught exception in JaxInterface::saveCheckpoint: " << ex.what();
    modelConditionVariable_.notify_all();
    throw;
  }
  modelConditionVariable_.notify_all();
}

void JaxInterface::loadCheckpoint(const std::string &modelCheckpointPath, const std::string &targetModelCheckpointPath, const std::string &optimizerStateCheckpointPath) {
  LOG(INFO) << "Loading checkpoint at paths \"" << modelCheckpointPath << "\", \"" << targetModelCheckpointPath << "\", and \"" << optimizerStateCheckpointPath << "\"";
  try {
    std::unique_lock modelLock(modelMutex_);
    modelConditionVariable_.wait(modelLock, [this]() -> bool {
      return !waitingToSelectAction_.load();
    });
    py::gil_scoped_acquire acquire;
    model_ = dqnModule_->attr("loadModelCheckpoint")(*model_, modelCheckpointPath);
    targetModel_ = dqnModule_->attr("loadModelCheckpoint")(*targetModel_, targetModelCheckpointPath);
    // Construct new optimizer for the loaded model.
    py::object optimizer = getPythonOptimizer();
    optimizerState_ = nnxModule_->attr("Optimizer")(*model_, optimizer);
    // Load the optimizer state into the newly created optimizer.
    optimizerState_ = dqnModule_->attr("loadOptimizerCheckpoint")(*optimizerState_, optimizerStateCheckpointPath);
  } catch (std::exception &ex) {
    LOG(ERROR) << "Caught exception in JaxInterface::loadCheckpoint: " << ex.what();
    modelConditionVariable_.notify_all();
    throw;
  }
  modelConditionVariable_.notify_all();
}

py::object JaxInterface::getPythonOptimizer() {
  return dqnModule_->attr("getOptaxAdamWOptimizer")(learningRate_);
}

py::object JaxInterface::getNextRngKey() {
  py::tuple keys = randomModule_->attr("split")(*rngKey_);
  if (keys.size() != 2) {
    throw std::runtime_error(absl::StrFormat("Tried to split key, but got back %d things", keys.size()));
  }
  rngKey_ = keys[0];
  return keys[1];
}

size_t JaxInterface::writeZerosToRawArray(size_t count, float *array) {
  std::fill_n(array, count, 0.0f);
  return count;
}

void JaxInterface::writeEmptyActionToRawArray(float *array) {
  std::fill_n(array, ActionSpace::size(), 0.0f);
}

void JaxInterface::writeActionToRawArray(int action, float *array) {
  std::fill_n(array, ActionSpace::size(), 0.0f);
  array[action] = 1.0f;
}

py::array_t<float> JaxInterface::createActionMaskNumpy(bool canSendPacket) {
  constexpr size_t kActionSpaceSize = ActionSpace::size();
  py::array_t<float> array(kActionSpaceSize);
  auto mutableArray = array.mutable_unchecked<1>();
  mutableArray(0) = 0.0; // IF-CHANGE: Sleep is supposed to be the first action in the action space.
  const float packetMaskValue = canSendPacket ? 0.0 : -std::numeric_limits<float>::infinity();
  std::fill_n(mutableArray.mutable_data(1), kActionSpaceSize-1, packetMaskValue);
  return array;
}

detail::ModelInputNumpy JaxInterface::modelInputToNumpy(const model_inputs::ModelInputView &modelInputView) {
  ZoneScopedN("JaxInterface::modelInputToNumpy");
  constexpr size_t kIndividualObservationSize = Observation::size();
  constexpr size_t kActionSpaceSize = ActionSpace::size();

  // Calculate the total size needed for past observations stack and current observation
  const size_t totalSize = kIndividualObservationSize * observationStackSize_;

  // ==================================== Allocate numpy arrays ====================================
  detail::ModelInputNumpy result;
  result.pastObservationStack      = py::array_t<float>({observationStackSize_, kIndividualObservationSize});
  result.pastObservationTimestamps = py::array_t<float>({observationStackSize_, size_t(1)});
  result.pastActions               = py::array_t<float>({observationStackSize_, kActionSpaceSize});
  result.pastMask                  = py::array_t<float>({observationStackSize_, size_t(1)});
  result.currentObservation        = py::array_t<float>(kIndividualObservationSize);

  // ============================ Write past observations to numpy array ===========================
  {
    auto mutablePastObservationStackArray = result.pastObservationStack.mutable_unchecked<2>();
    int pastObservationIndex = 0;
    // First write empty observations if the given observation stack does not fill the expected stack size.
    for (int i=modelInputView.pastObservationStack.size(); i<observationStackSize_; ++i) {
      float *observationDataPtr = mutablePastObservationStackArray.mutable_data(pastObservationIndex, 0);
      size_t written = writeZerosToRawArray(kIndividualObservationSize, observationDataPtr);
      if (written != kIndividualObservationSize) {
        LOG(WARNING) << "Wrote " << written << " but only expected to write " << kIndividualObservationSize;
      }
      ++pastObservationIndex;
    }
    // Now, write the past observations to the numpy array.
    for (int i=0; i<modelInputView.pastObservationStack.size(); ++i) {
      const Observation &observation = *modelInputView.pastObservationStack[i];
      float *observationDataPtr = mutablePastObservationStackArray.mutable_data(pastObservationIndex, 0);
      const size_t written = observation.writeToArray(observationDataPtr);
      if (written != kIndividualObservationSize) {
        LOG(WARNING) << "Wrote " << written << " but only expected to write " << kIndividualObservationSize;
      }
      ++pastObservationIndex;
    }
    if (pastObservationIndex != observationStackSize_) {
      LOG(WARNING) << "Wrote " << pastObservationIndex << " observations but expected to write " << observationStackSize_;
    }
  }

  // ======================= Write past observation timestamps to numpy array ======================
  {
    auto mutablePastObservationTimestampsArray = result.pastObservationTimestamps.mutable_unchecked<2>();
    // First write empty timestamps if the given observation stack does not fill the expected stack size.
    int pastObservationIndex = 0;
    for (int i=modelInputView.pastObservationStack.size(); i<observationStackSize_; ++i) {
      *mutablePastObservationTimestampsArray.mutable_data(pastObservationIndex, 0) = 0.0;
      ++pastObservationIndex;
    }
    // Now, write the past observation timestamps to the numpy array.
    for (int i=0; i<modelInputView.pastObservationStack.size(); ++i) {
      // Calculate the time delta between the current observation and the past observation.
      const int millisecondsAgo = std::chrono::duration_cast<std::chrono::milliseconds>(modelInputView.currentObservation->timestamp_ - modelInputView.pastObservationStack[i]->timestamp_).count();
      // For now, we hard-code the regularization target to be 2000ms.
      static constexpr int kRegularizationTargetMs = 2000;
      *mutablePastObservationTimestampsArray.mutable_data(pastObservationIndex, 0) = millisecondsAgo / static_cast<float>(kRegularizationTargetMs);
      ++pastObservationIndex;
    }
    if (pastObservationIndex != observationStackSize_) {
      LOG(WARNING) << "Wrote " << pastObservationIndex << " timestamps but expected to write " << observationStackSize_;
    }
  }

  // ============================== Write past actions to numpy array ==============================
  {
    auto mutablePastActionsArray = result.pastActions.mutable_unchecked<2>();
    // First write empty actions if the given observation stack does not fill the expected stack size.
    int pastActionIndex = 0;
    for (int i=modelInputView.pastObservationStack.size(); i<observationStackSize_; ++i) {
      float *actionDataPtr = mutablePastActionsArray.mutable_data(pastActionIndex, 0);
      writeEmptyActionToRawArray(actionDataPtr);
      ++pastActionIndex;
    }
    // Now, write the past actions to the numpy array.
    for (int i=0; i<modelInputView.pastObservationStack.size(); ++i) {
      float *actionDataPtr = mutablePastActionsArray.mutable_data(pastActionIndex, 0);
      writeActionToRawArray(modelInputView.pastActionStack[i], actionDataPtr);
      ++pastActionIndex;
    }
    if (pastActionIndex != observationStackSize_) {
      LOG(WARNING) << "Wrote " << pastActionIndex << " actions but expected to write " << observationStackSize_;
    }
  }

  // ================================ Write past mask to numpy array ===============================
  {
    auto mutablePastMaskArray = result.pastMask.mutable_unchecked<2>();
    // First write 0s if the given observation stack does not fill the expected stack size.
    int index = 0;
    for (int i=modelInputView.pastObservationStack.size(); i<observationStackSize_; ++i) {
      *mutablePastMaskArray.mutable_data(index, 0) = 0.0;
      ++index;
    }
    // Now, write 1s to the numpy array.
    for (int i=0; i<modelInputView.pastObservationStack.size(); ++i) {
      *mutablePastMaskArray.mutable_data(index, 0) = 1.0;
      ++index;
    }
    if (index != observationStackSize_) {
      LOG(WARNING) << "Wrote " << index << " masks but expected to write " << observationStackSize_;
    }
  }

  // =========================== Write current observation to numpy array ==========================
  auto mutableCurrentObservationArray = result.currentObservation.mutable_unchecked<1>();
  float *mutableCurrentObservationData = mutableCurrentObservationArray.mutable_data(0);
  const size_t written = modelInputView.currentObservation->writeToArray(mutableCurrentObservationData);
  if (written != kIndividualObservationSize) {
    LOG(WARNING) << "Wrote " << written << " but only expected to write " << kIndividualObservationSize;
  }

  return result;
}

detail::ModelInputNumpy JaxInterface::modelInputsToNumpy(const std::vector<model_inputs::ModelInputView> &modelInputViews) {
  ZoneScopedN("JaxInterface::modelInputsToNumpy");

  if (modelInputViews.empty()) {
    throw std::runtime_error("JaxInterface::modelInputsToNumpy: Batch size is 0");
  }

  constexpr size_t kIndividualObservationSize = Observation::size();
  constexpr size_t kActionSpaceSize = ActionSpace::size();

  // Calculate the total size needed for each model input (past observations stack + current observation)
  const size_t singleInputSize = kIndividualObservationSize * observationStackSize_;

  const size_t batchSize = modelInputViews.size();

  // ==================================== Allocate numpy arrays ====================================
  detail::ModelInputNumpy result;
  result.pastObservationStack      = py::array_t<float>({batchSize, observationStackSize_, kIndividualObservationSize});
  result.pastObservationTimestamps = py::array_t<float>({batchSize, observationStackSize_, size_t(1)});
  result.pastActions               = py::array_t<float>({batchSize, observationStackSize_, kActionSpaceSize});
  result.pastMask                  = py::array_t<float>({batchSize, observationStackSize_, size_t(1)});
  result.currentObservation        = py::array_t<float>({batchSize, kIndividualObservationSize});

  // ============================ Write past observations to numpy array ===========================
  for (int batchIndex=0; batchIndex<batchSize; ++batchIndex) {
    const model_inputs::ModelInputView &modelInput = modelInputViews[batchIndex];
    auto mutablePastObservationStackArray = result.pastObservationStack.mutable_unchecked<3>();
    int pastObservationIndex = 0;
    // First write empty observations if the given observation stack does not fill the expected stack size.
    for (int i=modelInput.pastObservationStack.size(); i<observationStackSize_; ++i) {
      float *observationDataPtr = mutablePastObservationStackArray.mutable_data(batchIndex, pastObservationIndex, 0);
      size_t written = writeZerosToRawArray(kIndividualObservationSize, observationDataPtr);
      if (written != kIndividualObservationSize) {
        LOG(WARNING) << "Wrote " << written << " but only expected to write " << kIndividualObservationSize;
      }
      ++pastObservationIndex;
    }
    // Now, write the past observations to the numpy array.
    for (int i=0; i<modelInput.pastObservationStack.size(); ++i) {
      const Observation &observation = *modelInput.pastObservationStack[i];
      float *observationDataPtr = mutablePastObservationStackArray.mutable_data(batchIndex, pastObservationIndex, 0);
      const size_t written = observation.writeToArray(observationDataPtr);
      if (written != kIndividualObservationSize) {
        LOG(WARNING) << "Wrote " << written << " but only expected to write " << kIndividualObservationSize;
      }
      ++pastObservationIndex;
    }
    if (pastObservationIndex != observationStackSize_) {
      LOG(WARNING) << "Wrote " << pastObservationIndex << " observations but expected to write " << observationStackSize_;
    }
  }

  // ======================= Write past observation timestamps to numpy array ======================
  for (int batchIndex=0; batchIndex<batchSize; ++batchIndex) {
    const model_inputs::ModelInputView &modelInput = modelInputViews[batchIndex];
    auto mutablePastObservationTimestampsArray = result.pastObservationTimestamps.mutable_unchecked<3>();
    // First write empty timestamps if the given observation stack does not fill the expected stack size.
    int pastObservationIndex = 0;
    for (int i=modelInput.pastObservationStack.size(); i<observationStackSize_; ++i) {
      *mutablePastObservationTimestampsArray.mutable_data(batchIndex, pastObservationIndex, 0) = 0.0;
      ++pastObservationIndex;
    }
    // Now, write the past observation timestamps to the numpy array.
    for (int i=0; i<modelInput.pastObservationStack.size(); ++i) {
      // Calculate the time delta between the current observation and the past observation.
      const int millisecondsAgo = std::chrono::duration_cast<std::chrono::milliseconds>(modelInput.currentObservation->timestamp_ - modelInput.pastObservationStack[i]->timestamp_).count();
      // For now, we hard-code the regularization target to be 2000ms.
      static constexpr int kRegularizationTargetMs = 2000;
      *mutablePastObservationTimestampsArray.mutable_data(batchIndex, pastObservationIndex, 0) = millisecondsAgo / static_cast<float>(kRegularizationTargetMs);
      ++pastObservationIndex;
    }
    if (pastObservationIndex != observationStackSize_) {
      LOG(WARNING) << "Wrote " << pastObservationIndex << " timestamps but expected to write " << observationStackSize_;
    }
  }

  // ============================== Write past actions to numpy array ==============================
  for (int batchIndex=0; batchIndex<batchSize; ++batchIndex) {
    const model_inputs::ModelInputView &modelInput = modelInputViews[batchIndex];
    auto mutablePastActionsArray = result.pastActions.mutable_unchecked<3>();
    // First write empty actions if the given observation stack does not fill the expected stack size.
    int pastActionIndex = 0;
    for (int i=modelInput.pastObservationStack.size(); i<observationStackSize_; ++i) {
      float *actionDataPtr = mutablePastActionsArray.mutable_data(batchIndex, pastActionIndex, 0);
      writeEmptyActionToRawArray(actionDataPtr);
      ++pastActionIndex;
    }
    // Now, write the past actions to the numpy array.
    for (int i=0; i<modelInput.pastObservationStack.size(); ++i) {
      float *actionDataPtr = mutablePastActionsArray.mutable_data(batchIndex, pastActionIndex, 0);
      writeActionToRawArray(modelInput.pastActionStack[i], actionDataPtr);
      ++pastActionIndex;
    }
    if (pastActionIndex != observationStackSize_) {
      LOG(WARNING) << "Wrote " << pastActionIndex << " actions but expected to write " << observationStackSize_;
    }
  }

  // ================================ Write past mask to numpy array ===============================
  for (int batchIndex=0; batchIndex<batchSize; ++batchIndex) {
    const model_inputs::ModelInputView &modelInput = modelInputViews[batchIndex];
    auto mutablePastMaskArray = result.pastMask.mutable_unchecked<3>();
    // First write 0s if the given observation stack does not fill the expected stack size.
    int index = 0;
    for (int i=modelInput.pastObservationStack.size(); i<observationStackSize_; ++i) {
      *mutablePastMaskArray.mutable_data(batchIndex, index, 0) = 0.0;
      ++index;
    }
    // Now, write 1s to the numpy array.
    for (int i=0; i<modelInput.pastObservationStack.size(); ++i) {
      *mutablePastMaskArray.mutable_data(batchIndex, index, 0) = 1.0;
      ++index;
    }
    if (index != observationStackSize_) {
      LOG(WARNING) << "Wrote " << index << " masks but expected to write " << observationStackSize_;
    }
  }

  // =========================== Write current observation to numpy array ==========================
  for (int batchIndex=0; batchIndex<batchSize; ++batchIndex) {
    const model_inputs::ModelInputView &modelInput = modelInputViews[batchIndex];
    auto mutableCurrentObservationArray = result.currentObservation.mutable_unchecked<2>();
    float *mutableCurrentObservationData = mutableCurrentObservationArray.mutable_data(batchIndex, 0);
    const size_t written = modelInput.currentObservation->writeToArray(mutableCurrentObservationData);
    if (written != kIndividualObservationSize) {
      LOG(WARNING) << "Wrote " << written << " but only expected to write " << kIndividualObservationSize;
    }
  }

  return result;
}

} // namespace rl