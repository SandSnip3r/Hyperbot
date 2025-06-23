#include "rl/jaxInterface.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <tracy/Tracy.hpp>

#include <absl/flags/flag.h>
#include <absl/log/log.h>
#include <absl/strings/str_join.h>
#include <absl/strings/str_format.h>

#include <algorithm>
#include <thread>

namespace py = pybind11;

ABSL_FLAG(bool, debug_nans, false, "Debug nan values");

namespace rl {

namespace {

bool checkHasAttr(py::object &obj, const std::string &attr_name) {
  // Get the hasattr function from Python's built-in module
  py::object hasattr_func = py::module_::import("builtins").attr("hasattr");

  // Call hasattr function
  return hasattr_func(obj, attr_name).cast<bool>();
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
  if (summaryWriter_.has_value()) {
    summaryWriter_.reset();
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

    // Initialize Tensorboard for logging statistics.
    py::module tensorboardX = py::module::import("tensorboardX");
    summaryWriter_ = tensorboardX.attr("SummaryWriter")("flush_secs"_a=1);

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

    const int observationSize = getObservationNumpySize(Observation());
    model_ = DqnModelType(observationSize, observationStackSize_, kActionSpaceSize, dropoutRate, *nnxRngs_);
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

JaxInterface::SelectActionOutput JaxInterface::selectAction(const model_inputs::ModelInputView &modelInputView,
                                                            bool canSendPacket) {
  ZoneScopedN("JaxInterface::selectAction");
  SelectActionOutput output;
  try {
    waitingToSelectAction_ = true;
    std::unique_lock lock(modelMutex_);
    waitingToSelectAction_ = false;
    py::gil_scoped_acquire acquire;
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
      // - rngKey
      actionPyObject = dqnModule_->attr("selectAction")(*model_,
                                                        numpyModelInput.pastObservationStack,
                                                        numpyModelInput.pastObservationTimestamps,
                                                        numpyModelInput.pastActions,
                                                        numpyModelInput.pastMask,
                                                        numpyModelInput.currentObservation,
                                                        actionMask);
    }
    py::tuple resultTuple = actionPyObject.cast<py::tuple>();
    output.actionIndex = resultTuple[0].cast<int>();
    output.qValues = resultTuple[1].cast<std::vector<float>>();
  } catch (std::exception &ex) {
    LOG(ERROR) << "Caught exception in JaxInterface::selectAction: " << ex.what();
    modelConditionVariable_.notify_all();
    throw;
  }
  modelConditionVariable_.notify_all();
  VLOG(1) << "Chose action " << output.actionIndex;
  return output;
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

void JaxInterface::addScalar(std::string_view name, double yValue, double xValue) {
  if (absl::GetFlag(FLAGS_debug_nans)) {
    if (std::isnan(yValue)) {
      LOG(ERROR) << "[" << name << "] yValue is nan: " << yValue;
    } else if (std::isinf(yValue)) {
      if (yValue > 0) {
        LOG(ERROR) << "[" << name << "] yValue is +inf: " << yValue;
      } else {
        LOG(ERROR) << "[" << name << "] yValue is -inf: " << yValue;
      }
    }
  }
  py::gil_scoped_acquire acquire;
  summaryWriter_->attr("add_scalar")(name, yValue, xValue);
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

size_t JaxInterface::getObservationNumpySize(const Observation &observation) const {
  return 6 + 1 + 23 +
      observation.remainingTimeOurBuffs_.size()*2 +
      observation.remainingTimeOpponentBuffs_.size()*2 +
      observation.remainingTimeOurDebuffs_.size()*2 +
      observation.remainingTimeOpponentDebuffs_.size()*2 +
      observation.skillCooldowns_.size()*2 +
      observation.itemCooldowns_.size()*2;
}

size_t JaxInterface::writeEmptyObservationToRawArray(size_t observationSize, float *array) {
  // Zero it out
  std::fill_n(array, observationSize, 0.0f);
  return observationSize;
}

#define LOG_IF_BEYOND_RANGE(value, min, max) \
do { \
  if (absl::GetFlag(FLAGS_debug_nans)) { \
    if (value < min) { \
      LOG(WARNING) << value << " is less than " << min; \
    } else if (value > max) { \
      LOG(WARNING) << value << " is greater than " << max; \
    } \
  } \
} while(false)

size_t JaxInterface::writeObservationToRawArray(const Observation &observation, float *array) {
  size_t index{0};
  array[index++] = observation.ourCurrentHp_ / static_cast<float>(observation.ourMaxHp_);
  LOG_IF_BEYOND_RANGE(array[index-1], 0.0, 1.0);
  array[index++] = observation.ourCurrentMp_ / static_cast<float>(observation.ourMaxMp_);
  LOG_IF_BEYOND_RANGE(array[index-1], 0.0, 1.0);
  array[index++] = observation.weAreKnockedDown_ ? 1.0 : 0.0;
  array[index++] = observation.opponentCurrentHp_ / static_cast<float>(observation.opponentMaxHp_);
  LOG_IF_BEYOND_RANGE(array[index-1], 0.0, 1.0);
  array[index++] = observation.opponentCurrentMp_ / static_cast<float>(observation.opponentMaxMp_);
  LOG_IF_BEYOND_RANGE(array[index-1], 0.0, 1.0);
  array[index++] = observation.opponentIsKnockedDown_ ? 1.0 : 0.0;

  array[index++] = observation.hpPotionCount_ / static_cast<float>(5); // IF-CHANGE: If we change this, also change TrainingManager::buildItemRequirementList
  LOG_IF_BEYOND_RANGE(array[index-1], 0.0, 1.0);
  index += writeOneHotEvent(observation.eventCode_, &array[index]);

  for (int i=0; i<observation.remainingTimeOurBuffs_.size(); ++i) {
    array[index++] = observation.remainingTimeOurBuffs_[i] != 0 ? 1.0 : 0.0;
    array[index++] = observation.remainingTimeOurBuffs_[i];
  }
  for (int i=0; i<observation.remainingTimeOpponentBuffs_.size(); ++i) {
    array[index++] = observation.remainingTimeOpponentBuffs_[i] != 0 ? 1.0 : 0.0;
    array[index++] = observation.remainingTimeOpponentBuffs_[i];
  }
  for (int i=0; i<observation.remainingTimeOurDebuffs_.size(); ++i) {
    array[index++] = observation.remainingTimeOurDebuffs_[i] != 0 ? 1.0 : 0.0;
    array[index++] = observation.remainingTimeOurDebuffs_[i];
  }
  for (int i=0; i<observation.remainingTimeOpponentDebuffs_.size(); ++i) {
    array[index++] = observation.remainingTimeOpponentDebuffs_[i] != 0 ? 1.0 : 0.0;
    array[index++] = observation.remainingTimeOpponentDebuffs_[i];
  }
  for (int cooldown : observation.skillCooldowns_) {
    array[index++] = cooldown == 0 ? 1.0 : 0.0;
    array[index++] = cooldown;
  }
  for (int cooldown : observation.itemCooldowns_) {
    array[index++] = cooldown == 0 ? 1.0 : 0.0;
    array[index++] = cooldown;
  }
  return index;
}

size_t JaxInterface::writeOneHotEvent(event::EventCode eventCode, float *array) {
  constexpr std::array kEventsWeCareAbout = {
    event::EventCode::kEntityBodyStateChanged,
    event::EventCode::kKnockdownStunEnded,
    event::EventCode::kKnockedDown,
    event::EventCode::kEntityPositionUpdated,
    event::EventCode::kEntityLifeStateChanged,
    event::EventCode::kEntityNotMovingAngleChanged,
    event::EventCode::kEntityMovementEnded,
    event::EventCode::kEntityMovementBegan,
    event::EventCode::kItemUseFailed,
    event::EventCode::kSkillCooldownEnded,
    event::EventCode::kEntityStatesChanged,
    event::EventCode::kBuffRemoved,
    event::EventCode::kBuffAdded,
    event::EventCode::kItemUseSuccess,
    event::EventCode::kItemCooldownEnded,
    event::EventCode::kItemMoved,
    event::EventCode::kSkillEnded,
    event::EventCode::kSkillBegan,
    event::EventCode::kEntityMpChanged,
    event::EventCode::kDealtDamage,
    event::EventCode::kEntityHpChanged,
    event::EventCode::kSkillFailed,
    event::EventCode::kCommandError
  };
  std::fill_n(array, kEventsWeCareAbout.size(), 0.0f);
  const auto it = std::find(kEventsWeCareAbout.begin(), kEventsWeCareAbout.end(), eventCode);
  if (it != kEventsWeCareAbout.end()) {
    array[std::distance(kEventsWeCareAbout.begin(), it)] = 1.0f;
  } else {
    // If the given event is not in the list, no event will be set.
  }
  return kEventsWeCareAbout.size();
}

void JaxInterface::writeEmptyActionToRawArray(float *array) {
  std::fill_n(array, kActionSpaceSize, 0.0f);
}

void JaxInterface::writeActionToRawArray(int action, float *array) {
  std::fill_n(array, kActionSpaceSize, 0.0f);
  array[action] = 1.0f;
}

py::array_t<float> JaxInterface::createActionMaskNumpy(bool canSendPacket) {
  py::array_t<float> array(kActionSpaceSize);
  auto mutableArray = array.mutable_unchecked<1>();
  mutableArray(0) = 0.0;
  const float packetMaskValue = canSendPacket ? 0.0 : -std::numeric_limits<float>::infinity();
  std::fill_n(mutableArray.mutable_data(1), kActionSpaceSize-1, packetMaskValue);
  return array;
}

detail::ModelInputNumpy JaxInterface::modelInputToNumpy(const model_inputs::ModelInputView &modelInputView) {
  ZoneScopedN("JaxInterface::modelInputToNumpy");

  // Get the size of an observation
  const size_t individualObservationSize = getObservationNumpySize(*modelInputView.currentObservation);

  // Calculate the total size needed for past observations stack and current observation
  const size_t totalSize = individualObservationSize * observationStackSize_;

  // ==================================== Allocate numpy arrays ====================================
  detail::ModelInputNumpy result;
  result.pastObservationStack      = py::array_t<float>({observationStackSize_, individualObservationSize});
  result.pastObservationTimestamps = py::array_t<float>({observationStackSize_, size_t(1)});
  result.pastActions               = py::array_t<float>({observationStackSize_, kActionSpaceSize});
  result.pastMask                  = py::array_t<float>({observationStackSize_, size_t(1)});
  result.currentObservation        = py::array_t<float>(individualObservationSize);

  // ============================ Write past observations to numpy array ===========================
  {
    auto mutablePastObservationStackArray = result.pastObservationStack.mutable_unchecked<2>();
    int pastObservationIndex = 0;
    // First write empty observations if the given observation stack does not fill the expected stack size.
    for (int i=modelInputView.pastObservationStack.size(); i<observationStackSize_; ++i) {
      float *observationDataPtr = mutablePastObservationStackArray.mutable_data(pastObservationIndex, 0);
      size_t written = writeEmptyObservationToRawArray(individualObservationSize, observationDataPtr);
      if (written != individualObservationSize) {
        LOG(WARNING) << "Wrote " << written << " but only expected to write " << individualObservationSize;
      }
      ++pastObservationIndex;
    }
    // Now, write the past observations to the numpy array.
    for (int i=0; i<modelInputView.pastObservationStack.size(); ++i) {
      const Observation &observation = *modelInputView.pastObservationStack[i];
      float *observationDataPtr = mutablePastObservationStackArray.mutable_data(pastObservationIndex, 0);
      size_t written = writeObservationToRawArray(observation, observationDataPtr);
      if (written != individualObservationSize) {
        LOG(WARNING) << "Wrote " << written << " but only expected to write " << individualObservationSize;
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
  size_t written = writeObservationToRawArray(*modelInputView.currentObservation, mutableCurrentObservationData);
  if (written != individualObservationSize) {
    LOG(WARNING) << "Wrote " << written << " but only expected to write " << individualObservationSize;
  }

  return result;
}

detail::ModelInputNumpy JaxInterface::modelInputsToNumpy(const std::vector<model_inputs::ModelInputView> &modelInputViews) {
  ZoneScopedN("JaxInterface::modelInputsToNumpy");

  if (modelInputViews.empty()) {
    throw std::runtime_error("JaxInterface::modelInputsToNumpy: Batch size is 0");
  }

  // Get the size of a single observation
  const size_t individualObservationSize = getObservationNumpySize(*modelInputViews[0].currentObservation);

  // Calculate the total size needed for each model input (past observations stack + current observation)
  const size_t singleInputSize = individualObservationSize * observationStackSize_;

  const size_t batchSize = modelInputViews.size();

  // ==================================== Allocate numpy arrays ====================================
  detail::ModelInputNumpy result;
  result.pastObservationStack      = py::array_t<float>({batchSize, observationStackSize_, individualObservationSize});
  result.pastObservationTimestamps = py::array_t<float>({batchSize, observationStackSize_, size_t(1)});
  result.pastActions               = py::array_t<float>({batchSize, observationStackSize_, kActionSpaceSize});
  result.pastMask                  = py::array_t<float>({batchSize, observationStackSize_, size_t(1)});
  result.currentObservation        = py::array_t<float>({batchSize, individualObservationSize});

  // ============================ Write past observations to numpy array ===========================
  for (int batchIndex=0; batchIndex<batchSize; ++batchIndex) {
    const model_inputs::ModelInputView &modelInput = modelInputViews[batchIndex];
    auto mutablePastObservationStackArray = result.pastObservationStack.mutable_unchecked<3>();
    int pastObservationIndex = 0;
    // First write empty observations if the given observation stack does not fill the expected stack size.
    for (int i=modelInput.pastObservationStack.size(); i<observationStackSize_; ++i) {
      float *observationDataPtr = mutablePastObservationStackArray.mutable_data(batchIndex, pastObservationIndex, 0);
      size_t written = writeEmptyObservationToRawArray(individualObservationSize, observationDataPtr);
      if (written != individualObservationSize) {
        LOG(WARNING) << "Wrote " << written << " but only expected to write " << individualObservationSize;
      }
      ++pastObservationIndex;
    }
    // Now, write the past observations to the numpy array.
    for (int i=0; i<modelInput.pastObservationStack.size(); ++i) {
      const Observation &observation = *modelInput.pastObservationStack[i];
      float *observationDataPtr = mutablePastObservationStackArray.mutable_data(batchIndex, pastObservationIndex, 0);
      size_t written = writeObservationToRawArray(observation, observationDataPtr);
      if (written != individualObservationSize) {
        LOG(WARNING) << "Wrote " << written << " but only expected to write " << individualObservationSize;
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
    size_t written = writeObservationToRawArray(*modelInput.currentObservation, mutableCurrentObservationData);
    if (written != individualObservationSize) {
      LOG(WARNING) << "Wrote " << written << " but only expected to write " << individualObservationSize;
    }
  }

  return result;
}

} // namespace rl