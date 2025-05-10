#include "rl/jaxInterface.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <tracy/Tracy.hpp>

#include <absl/log/log.h>
#include <absl/strings/str_join.h>
#include <absl/strings/str_format.h>

#include <algorithm>
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

void JaxInterface::initialize() {
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

    const int kInputSize = observationStackSize_ * getObservationNumpySize(Observation());
    const int kOutputSize = kActionSpaceSize;
    model_ = DqnModelType(kInputSize, kOutputSize, *nnxRngs_);
    targetModel_ = py::module::import("copy").attr("deepcopy")(*model_);

    optaxModule_ = py::module::import("optax");
    py::object optimizer = getOptimizer();
    optimizerState_ = nnxModule_->attr("Optimizer")(*model_, optimizer);
  }
  modelConditionVariable_.notify_all();
}

int JaxInterface::selectAction(const ModelInput &modelInput, bool canSendPacket) {
  ZoneScopedN("JaxInterface::selectAction");
  int actionIndex;
  try {
    waitingToSelectAction_ = true;
    std::unique_lock lock(modelMutex_);
    waitingToSelectAction_ = false;
    py::gil_scoped_acquire acquire;
    // Convert C++ observation into numpy observation
    py::array_t<float> numpyObservation = modelInputToNumpy(modelInput);
    // Create an action mask based on whether or not we can send a packet
    py::object actionMask = createActionMask(canSendPacket);
    // Get the action from the model
    py::object actionPyObject;
    {
      ZoneScopedN("JaxInterface::selectAction_PYTHON");
      actionPyObject = dqnModule_->attr("selectAction")(*model_, numpyObservation, actionMask, getNextRngKey());
    }
    actionIndex = actionPyObject.cast<int>();
  } catch (std::exception &ex) {
    LOG(ERROR) << "Caught exception in JaxInterface::selectAction: " << ex.what();
    modelConditionVariable_.notify_all();
    throw;
  }
  modelConditionVariable_.notify_all();
  VLOG(1) << "Chose action " << actionIndex;
  return actionIndex;
}

JaxInterface::TrainAuxOutput JaxInterface::train(const std::vector<ModelInput> &oldModelInputs,
                                                 const std::vector<int> &actionsTaken,
                                                 const std::vector<bool> &isTerminals,
                                                 const std::vector<float> &rewards,
                                                 const std::vector<ModelInput> &newModelInputs,
                                                 const std::vector<float> &importanceSamplingWeights) {
  ZoneScopedN("JaxInterface::train");
  {
    size_t size = oldModelInputs.size();
    if (size == 0 ||
        actionsTaken.size() != size ||
        isTerminals.size() != size ||
        rewards.size() != size ||
        newModelInputs.size() != size ||
        importanceSamplingWeights.size() != size) {
      throw std::runtime_error(absl::StrFormat("JaxInterface::train: Batch size mismatch: %zu %zu %zu %zu %zu %zu",
        oldModelInputs.size(), actionsTaken.size(), isTerminals.size(), rewards.size(), newModelInputs.size(), importanceSamplingWeights.size()));
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
      py::array_t<float> oldModelInputNumpy = modelInputsToNumpy(oldModelInputs);
      py::array_t<float> newModelInputNumpy = modelInputsToNumpy(newModelInputs);
      py::object auxOutput;
      {
        ZoneScopedN("JaxInterface::train_PYTHON");
        auxOutput = dqnModule_->attr("jittedTrain")(*model_, *optimizerState_, *targetModel_,
                                                    oldModelInputNumpy, actionsTaken, isTerminals,
                                                    rewards, newModelInputNumpy, importanceSamplingWeights, gamma_);
      }
      py::tuple auxOutputTuple = auxOutput.cast<py::tuple>();
      result.tdErrors = auxOutputTuple[0].cast<std::vector<float>>();
      result.meanMinQValue = auxOutputTuple[1].cast<float>();
      result.meanMeanQValue = auxOutputTuple[2].cast<float>();
      result.meanMaxQValue = auxOutputTuple[3].cast<float>();
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
    py::object optimizer = getOptimizer();
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
  py::gil_scoped_acquire acquire;
  summaryWriter_->attr("add_scalar")(name, yValue, xValue);
}

py::object JaxInterface::getOptimizer() {
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

py::object JaxInterface::observationToNumpy(const Observation &observation) {
  ZoneScopedN("JaxInterface::observationToNumpy");
  py::array_t<float> array(getObservationNumpySize(observation));
  auto mutableArray = array.mutable_unchecked<1>();
  float *mutableData = mutableArray.mutable_data(0);
  writeObservationToRawArray(observation, mutableData);
  return array;
}

py::object JaxInterface::observationStackToNumpy(int stackSize, const std::vector<Observation> &observationStack) {
  ZoneScopedN("JaxInterface::observationStackToNumpy");
  if (observationStack.empty()) {
    throw std::runtime_error("JaxInterface::observationStackToNumpy: empty observation stack");
  }
  // Allocate a 1d numpy array to hold a flattening of all observations.
  const size_t observationSize = getObservationNumpySize(observationStack.at(0));
  py::array_t<float> array(stackSize * observationSize);
  // Grab a raw pointer to the numpy array.
  auto mutableArray = array.mutable_unchecked<1>();
  float *mutableData = mutableArray.mutable_data(0);
  int index{0};
  // First, write empty observations if the given observation stack does not fill the expected stack size.
  for (int i=observationStack.size(); i<stackSize; ++i) {
    writeEmptyObservationToRawArray(observationSize, mutableData + index*observationSize);
    ++index;
  }
  // Write the actual observations to the numpy array.
  for (const Observation &observation : observationStack) {
    writeObservationToRawArray(observation, mutableData + index*observationSize);
    ++index;
  }
  return array;
}

py::object JaxInterface::observationsToNumpy(const std::vector<Observation> &observations) {
  ZoneScopedN("JaxInterface::observationsToNumpy");
  py::array_t<float> array({observations.size(), getObservationNumpySize(observations.at(0))});
  auto mutableArray = array.mutable_unchecked<2>();
  for (size_t batchIndex=0; batchIndex<observations.size(); ++batchIndex) {
    const Observation &observation = observations.at(batchIndex);
    float *mutableData = mutableArray.mutable_data(batchIndex, 0);
    writeObservationToRawArray(observation, mutableData);
  }
  return array;
}

py::object JaxInterface::observationStacksToNumpy(int stackSize, const std::vector<std::vector<Observation>> &observationStacks) {
  ZoneScopedN("JaxInterface::observationStacksToNumpy");
  if (observationStacks.empty()) {
    throw std::runtime_error("JaxInterface::observationStacksToNumpy: Batch size is 0");
  }
  for (const auto &observationStack : observationStacks) {
    if (observationStack.empty()) {
      throw std::runtime_error("JaxInterface::observationStacksToNumpy: empty observation stack");
    }
  }
  // Allocate a 2d numpy array to hold a batch of a flattening of all observations.
  const size_t observationSize = getObservationNumpySize(observationStacks.at(0).at(0));
  py::array_t<float> array({observationStacks.size(), stackSize * observationSize});
  // Grab a raw pointer to the numpy array.
  auto mutableArray = array.mutable_unchecked<2>();
  for (size_t batchIndex=0; batchIndex<observationStacks.size(); ++batchIndex) {
    const std::vector<Observation> &observationStack = observationStacks.at(batchIndex);
    float *mutableData = mutableArray.mutable_data(batchIndex, 0);
    int index{0};
    // First, write empty observations if the given observation stack does not fill the expected stack size.
    for (int i=observationStack.size(); i<stackSize; ++i) {
      writeEmptyObservationToRawArray(observationSize, mutableData + index*observationSize);
      ++index;
    }
    // Write the actual observations to the numpy array.
    for (const Observation &observation : observationStack) {
      writeObservationToRawArray(observation, mutableData + index*observationSize);
      ++index;
    }
  }
  return array;
}

size_t JaxInterface::getObservationNumpySize(const Observation &observation) const {
  return 6 + 1 + 23 + observation.remainingTimeOurBuffs_.size()*2 + observation.remainingTimeOpponentBuffs_.size()*2 + observation.skillCooldowns_.size()*2 + observation.itemCooldowns_.size()*2;
}

size_t JaxInterface::writeEmptyObservationToRawArray(int observationSize, float *array) {
  // Zero it out
  std::fill_n(array, observationSize, 0.0f);
  return observationSize;
}

size_t JaxInterface::writeObservationToRawArray(const Observation &observation, float *array) {
  size_t index{0};
  array[index++] = observation.ourCurrentHp_ / static_cast<float>(observation.ourMaxHp_);
  array[index++] = observation.ourCurrentMp_ / static_cast<float>(observation.ourMaxMp_);
  array[index++] = observation.weAreKnockedDown_ ? 1.0 : 0.0;
  array[index++] = observation.opponentCurrentHp_ / static_cast<float>(observation.opponentMaxHp_);
  array[index++] = observation.opponentCurrentMp_ / static_cast<float>(observation.opponentMaxMp_);
  array[index++] = observation.opponentIsKnockedDown_ ? 1.0 : 0.0;

  array[index++] = observation.hpPotionCount_ / static_cast<float>(5); // IF-CHANGE: If we change this, also change TrainingManager::buildItemRequirementList

  index += writeOneHotEvent(observation.eventCode_, &array[index]);

  for (int i=0; i<observation.remainingTimeOurBuffs_.size(); ++i) {
    array[index++] = observation.remainingTimeOurBuffs_[i] != 0 ? 1.0 : 0.0;
    array[index++] = static_cast<float>(observation.remainingTimeOurBuffs_[i]) / 1000.0; // Convert milliseconds to seconds
  }
  for (int i=0; i<observation.remainingTimeOpponentBuffs_.size(); ++i) {
    array[index++] = observation.remainingTimeOpponentBuffs_[i] != 0 ? 1.0 : 0.0;
    array[index++] = static_cast<float>(observation.remainingTimeOpponentBuffs_[i]) / 1000.0; // Convert milliseconds to seconds
  }
  for (int cooldown : observation.skillCooldowns_) {
    array[index++] = cooldown == 0 ? 1.0 : 0.0;
    array[index++] = static_cast<float>(cooldown) / 1000.0; // Convert milliseconds to seconds
  }
  for (int cooldown : observation.itemCooldowns_) {
    array[index++] = cooldown == 0 ? 1.0 : 0.0;
    array[index++] = static_cast<float>(cooldown) / 1000.0; // Convert milliseconds to seconds
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
    event::EventCode::kPlayerCharacterBuffRemoved,
    event::EventCode::kPlayerCharacterBuffAdded,
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

pybind11::object JaxInterface::createActionMask(bool canSendPacket) {
  py::array_t<float> array(kActionSpaceSize);
  auto mutableArray = array.mutable_unchecked<1>();
  mutableArray(0) = 0.0;
  const float packetMaskValue = canSendPacket ? 0.0 : -std::numeric_limits<float>::infinity();
  std::fill_n(mutableArray.mutable_data(1), kActionSpaceSize-1, packetMaskValue);
  return array;
}

py::array_t<float> JaxInterface::modelInputToNumpy(const ModelInput &modelInput) {
  ZoneScopedN("JaxInterface::modelInputToNumpy");

  // Get the size of an observation
  const size_t individualObservationSize = getObservationNumpySize(*modelInput.currentObservation);

  // Calculate the total size needed for past observations stack and current observation
  const size_t totalSize = individualObservationSize * observationStackSize_;

  // Allocate a 1d numpy array to hold a flattening of all observations
  py::array_t<float> array(totalSize);

  // Grab a raw pointer to the numpy array
  auto mutableArray = array.mutable_unchecked<1>();
  float *mutableData = mutableArray.mutable_data(0);

  size_t floatsWritten = writeModelInputToRawArray(modelInput, mutableData);
  if (floatsWritten != totalSize) {
    throw std::runtime_error(absl::StrFormat("JaxInterface::modelInputToNumpy: writeModelInputToRawArray wrote an unexpected number of floats. %d != %d", floatsWritten, totalSize));
  }

  return array;
}

py::array_t<float> JaxInterface::modelInputsToNumpy(const std::vector<ModelInput> &modelInputs) {
  ZoneScopedN("JaxInterface::modelInputsToNumpy");

  if (modelInputs.empty()) {
    throw std::runtime_error("JaxInterface::modelInputsToNumpy: Batch size is 0");
  }

  // Get the size of a single observation
  const size_t individualObservationSize = getObservationNumpySize(*modelInputs[0].currentObservation);

  // Calculate the total size needed for each model input (past observations stack + current observation)
  const size_t singleInputSize = individualObservationSize * observationStackSize_;

  // Allocate a 2d numpy array to hold a batch of model inputs
  py::array_t<float> array({modelInputs.size(), singleInputSize});

  // Grab a raw pointer to the numpy array
  auto mutableArray = array.mutable_unchecked<2>();

  // Process each model input in the batch
  for (size_t batchIndex = 0; batchIndex < modelInputs.size(); ++batchIndex) {
    const ModelInput &modelInput = modelInputs[batchIndex];
    float *mutableData = mutableArray.mutable_data(batchIndex, 0);

    size_t floatsWritten = writeModelInputToRawArray(modelInput, mutableData);
    if (floatsWritten != individualObservationSize * observationStackSize_) {
      throw std::runtime_error(absl::StrFormat("JaxInterface::modelInputsToNumpy: writeModelInputToRawArray wrote an unexpected number of floats. %d != %d", floatsWritten, individualObservationSize * observationStackSize_));
    }
  }

  return array;
}

size_t JaxInterface::writeModelInputToRawArray(const ModelInput &modelInput, float *array) {
  const size_t individualObservationSize = getObservationNumpySize(*modelInput.currentObservation);
  size_t index = 0;

  int emptyCount=0;
  int nonEmptyCount=0;

  // Write empty observations if the given observation stack does not fill the expected stack size.
  for (int i=modelInput.pastObservationStack.size()+1; i<observationStackSize_; ++i) {
    size_t written = writeEmptyObservationToRawArray(individualObservationSize, array+index);
    if (written != individualObservationSize) {
      LOG(WARNING) << "Wrote " << written << " but only expected to write " << individualObservationSize;
    }
    index += individualObservationSize;
    ++emptyCount;
  }

  // Write the past observations to the numpy array
  for (const Observation *observation : modelInput.pastObservationStack) {
    size_t written = writeObservationToRawArray(*observation, array+index);
    if (written != individualObservationSize) {
      LOG(WARNING) << "Wrote " << written << " but only expected to write " << individualObservationSize;
    }
    index += individualObservationSize;
    ++nonEmptyCount;
  }

  // Write the current observation to the end of the array
  size_t written = writeObservationToRawArray(*modelInput.currentObservation, array+index);
  if (written != individualObservationSize) {
    LOG(WARNING) << "Wrote " << written << " but only expected to write " << individualObservationSize;
  }
  index += individualObservationSize;

  return index;
}

} // namespace rl