#include "rl/jaxInterface.hpp"

#include <pybind11/numpy.h>

#include <tracy/Tracy.hpp>

#include <absl/log/log.h>
#include <absl/strings/str_format.h>

#include <thread>

namespace py = pybind11;

namespace rl {

JaxInterface::~JaxInterface() {
  VLOG(1) << "Destructing JaxInterface";
  py::gil_scoped_acquire acquire;
  if (jaxModule_.has_value()) {
    jaxModule_.reset();
  }
  if (randomModule_.has_value()) {
    randomModule_.reset();
  }
  if (nnxModule_.has_value()) {
    nnxModule_.reset();
  }
  if (rngKey_.has_value()) {
    rngKey_.reset();
  }
  if (nnxRngs_.has_value()) {
    nnxRngs_.reset();
  }
  if (modelGraph_.has_value()) {
    modelGraph_.reset();
  }
  if (modelWeights_.has_value()) {
    modelWeights_.reset();
  }
}

void JaxInterface::initialize() {
  VLOG(1) << "Constructing JaxInterface";
  py::gil_scoped_acquire acquire;
  py::object MyModelType;
  py::object myModel;
  py::tuple graphAndWeights;

  jaxModule_ = py::module::import("rl.python.myPython");
  randomModule_ = py::module::import("jax.random");
  // Grab a random key based on our seed. Any randomness from this point on will split & replace this key held in member data.
  rngKey_ = randomModule_->attr("key")(kSeed);
  // NNX's Rngs is created using a JAX key, so we'll use the above key to create our NNX Rngs.
  nnxModule_ = py::module::import("flax.nnx");
  nnxRngs_ = nnxModule_->attr("Rngs")(getNextRngKey());
  // Now, we want to create a randomly initialized model. Specifically, we want randomly initialized weights. To do this, we'll instantiate our NNX model, then split the abstract graph and the concrete weights.
  MyModelType = jaxModule_->attr("MyModel");
  const int kInputSize = 4 + 32*2 + 3*2;
  const int kOutputSize = 38;
  myModel = MyModelType(kInputSize, kOutputSize, *nnxRngs_);
  graphAndWeights = nnxModule_->attr("split")(myModel);
  modelGraph_ = graphAndWeights[0];
  modelWeights_ = graphAndWeights[1];
}

void JaxInterface::train() {
  LOG(INFO) << "All aboard the JAX train!";
  py::gil_scoped_acquire acquire;
  jaxModule_->attr("func")(12345);
}

int JaxInterface::selectAction(const Observation &observation) {
  ZoneScopedN("JaxInterface::selectAction");
  VLOG(1) << "Getting action for observation " << observation.toString();
  py::gil_scoped_acquire acquire;
  // Convert C++ observation into numpy observation
  py::object numpyObservation = convertToNumpy(observation);
  // Pick the right weights for the model
  py::object model = nnxModule_->attr("merge")(*modelGraph_, *modelWeights_);
  // Get the action from the model
  py::object actionPyObject;
  {
    ZoneScopedN("JaxInterface::selectAction_PYTHON");
    actionPyObject = jaxModule_->attr("selectAction")(model, numpyObservation, getNextRngKey());
  }
  int actionIndex = actionPyObject.cast<int>();
  VLOG(1) << "Chose action " << actionIndex;
  return actionIndex;
}

py::object JaxInterface::getNextRngKey() {
  py::tuple keys = randomModule_->attr("split")(*rngKey_);
  if (keys.size() != 2) {
    throw std::runtime_error(absl::StrFormat("Tried to split key, but got back %d things", keys.size()));
  }
  rngKey_ = keys[0];
  return keys[1];
}

py::object JaxInterface::convertToNumpy(const Observation &observation) {
  ZoneScopedN("JaxInterface::convertToNumpy");
  py::array_t<float> array(4 + observation.skillCooldowns_.size()*2 + observation.itemCooldowns_.size()*2);
  auto mutableArray = array.mutable_unchecked<1>();
  mutableArray(0) = observation.ourCurrentHp_ / static_cast<float>(observation.ourMaxHp_);
  mutableArray(1) = observation.ourCurrentMp_ / static_cast<float>(observation.ourMaxMp_);
  mutableArray(2) = observation.opponentCurrentHp_ / static_cast<float>(observation.opponentMaxHp_);
  mutableArray(3) = observation.opponentCurrentMp_ / static_cast<float>(observation.opponentMaxMp_);
  int index = 4;
  for (int cooldown : observation.skillCooldowns_) {
    mutableArray(index) = cooldown == 0 ? 1.0 : 0.0;
    mutableArray(index+1) = static_cast<float>(cooldown) / 1000.0;
    index += 2;
  }
  for (int cooldown : observation.itemCooldowns_) {
    mutableArray(index) = cooldown == 0 ? 1.0 : 0.0;
    mutableArray(index+1) = static_cast<float>(cooldown) / 1000.0;
    index += 2;
  }
  return array;
}

} // namespace rl