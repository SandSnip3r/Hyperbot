#include "rl/jaxInterface.hpp"

#include <pybind11/numpy.h>

#include <tracy/Tracy.hpp>

#include <absl/log/log.h>
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
    const int kInputSize = 4 + 1 + 32*2 + 3*2; // IF-CHANGE: If we change this, also change JaxInterface::observationToNumpy
    const int kOutputSize = kActionSpaceSize;
    model_ = DqnModelType(kInputSize, kOutputSize, *nnxRngs_);
    targetModel_ = py::module::import("copy").attr("deepcopy")(*model_);

    optaxModule_ = py::module::import("optax");
    py::object adam = optaxModule_->attr("adam")(kLearningRate);
    optimizerState_ = nnxModule_->attr("Optimizer")(*model_, adam);
  }
  modelConditionVariable_.notify_all();
}

int JaxInterface::selectAction(const Observation &observation, bool canSendPacket) {
  ZoneScopedN("JaxInterface::selectAction");
  VLOG(1) << absl::StreamFormat("Getting action for observation %s, canSendPacket=%v", observation.toString(), canSendPacket);
  int actionIndex;
  try {
    waitingToSelectAction_ = true;
    std::unique_lock lock(modelMutex_);
    waitingToSelectAction_ = false;
    py::gil_scoped_acquire acquire;
    // Convert C++ observation into numpy observation
    py::object numpyObservation = observationToNumpy(observation);
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

JaxInterface::TrainAuxOutput JaxInterface::train(const Observation &olderObservation, int actionIndex, bool isTerminal, float reward, const Observation &newerObservation, float weight) {
  ZoneScopedN("JaxInterface::train");
  {
    std::unique_lock modelLock(modelMutex_);
    modelConditionVariable_.wait(modelLock, [this]() -> bool {
      return !waitingToSelectAction_.load();
    });
    py::gil_scoped_acquire acquire;
    try {
      ZoneScopedN("JaxInterface::train_PYTHON");
      py::object auxOutput = dqnModule_->attr("train")(*model_, *optimizerState_, *targetModel_, observationToNumpy(olderObservation), actionIndex, isTerminal, reward, observationToNumpy(newerObservation), weight);
      py::tuple auxOutputTuple = auxOutput.cast<py::tuple>();
      TrainAuxOutput result;
      result.tdError = auxOutputTuple[0].cast<float>();
      result.minQValue = auxOutputTuple[1].cast<float>();
      result.meanQValue = auxOutputTuple[2].cast<float>();
      result.maxQValue = auxOutputTuple[3].cast<float>();
      return result;
    } catch (std::exception &ex) {
      LOG(ERROR) << "Caught exception in JaxInterface::train: " << ex.what();
      throw;
    }
  }
  modelConditionVariable_.notify_all();
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
    py::object adam = optaxModule_->attr("adam")(kLearningRate);
    optimizerState_ = nnxModule_->attr("Optimizer")(*model_, adam);
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
  // IF-CHANGE: If we change this, also change JaxInterface::initialize of DqnModel
  py::array_t<float> array(4 + 1 + observation.skillCooldowns_.size()*2 + observation.itemCooldowns_.size()*2);
  auto mutableArray = array.mutable_unchecked<1>();
  int index{0};
  mutableArray(index++) = observation.ourCurrentHp_ / static_cast<float>(observation.ourMaxHp_);
  mutableArray(index++) = observation.ourCurrentMp_ / static_cast<float>(observation.ourMaxMp_);
  mutableArray(index++) = observation.opponentCurrentHp_ / static_cast<float>(observation.opponentMaxHp_);
  mutableArray(index++) = observation.opponentCurrentMp_ / static_cast<float>(observation.opponentMaxMp_);
  mutableArray(index++) = observation.hpPotionCount_ / static_cast<float>(5); // IF-CHANGE: If we change this, also change TrainingManager::buildItemRequirementList
  for (int cooldown : observation.skillCooldowns_) {
    mutableArray(index) = cooldown == 0 ? 1.0 : 0.0;
    mutableArray(index+1) = static_cast<float>(cooldown) / 1000.0; // Convert milliseconds to seconds
    index += 2;
  }
  for (int cooldown : observation.itemCooldowns_) {
    mutableArray(index) = cooldown == 0 ? 1.0 : 0.0;
    mutableArray(index+1) = static_cast<float>(cooldown) / 1000.0; // Convert milliseconds to seconds
    index += 2;
  }
  return array;
}

pybind11::object JaxInterface::createActionMask(bool canSendPacket) {
  py::array_t<float> array(kActionSpaceSize);
  auto mutableArray = array.mutable_unchecked<1>();
  mutableArray(0) = 0.0;
  const float packetMaskValue = canSendPacket ? 0.0 : -std::numeric_limits<float>::infinity();
  std::fill_n(mutableArray.mutable_data(1), kActionSpaceSize-1, packetMaskValue);
  return array;
}

} // namespace rl