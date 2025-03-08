#include "rl/jaxInterface.hpp"

#include <pybind11/numpy.h>

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
  try {
    py::module nnxModule;
    py::object MyModelType;
    py::object myModel;
    py::tuple graphAndWeights;

    LOG(INFO) << "Begin python area";
    jaxModule_ = py::module::import("rl.python.myPython");
    // LOG(INFO) << 'a';
    // randomModule_ = py::module::import("jax.random");
    // LOG(INFO) << 'a';
    // // Grab a random key based on our seed. Any randomness from this point on will split & replace this key held in member data.
    // rngKey_ = randomModule_->attr("key")(kSeed);
    // LOG(INFO) << 'a';
    // // NNX's Rngs is created using a JAX key, so we'll use the above key to create our NNX Rngs.
    // nnxModule = py::module::import("flax.nnx");
    // LOG(INFO) << 'a';
    // nnxRngs_ = nnxModule.attr("Rngs")(getNextRngKey()); // Error here
    // LOG(INFO) << 'a';
    // // Now, we want to create a randomly initialized model. Specifically, we want randomly initialized weights. To do this, we'll instantiate our NNX model, then split the abstract graph and the concrete weights.

    // MyModelType = jaxModule_->attr("MyModel");
    // LOG(INFO) << 'a';
    // const int kInputSize = 2;
    // const int kOutputSize = 38;
    // myModel = MyModelType(kInputSize, kOutputSize, nnxRngs_);
    // LOG(INFO) << 'a';
    // graphAndWeights = nnxModule.attr("split")(myModel);
    // LOG(INFO) << 'a';
    // modelGraph_ = graphAndWeights[0];
    // LOG(INFO) << 'a';
    // modelWeights_ = graphAndWeights[1];
    // LOG(INFO) << 'a';
  } catch (...) {
    LOG(ERROR) << "Failed to construct JaxInterface";
  }
}

void JaxInterface::train() {
  LOG(INFO) << "All aboard the JAX train!";
  {
    py::gil_scoped_acquire acquire;
    jaxModule_->attr("func")(12345);
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(5000));
}

int JaxInterface::selectAction(const Observation &observation) {
  // LOG(INFO) << "Getting action for observation";
  // py::gil_scoped_acquire acquire;
  // // Convert C++ observation into numpy observation
  // py::object numpyObservation = convertToNumpy(observation);
  // // const py::object actionPyObject = jaxModule_.attr("selectAction")(numpyObservation, getNextRngKey());
  // const py::object actionPyObject = jaxModule_->attr("selectAction")(123, getNextRngKey());
  // return actionPyObject.cast<int>();
  return 0;
}

py::object JaxInterface::getNextRngKey() {
  py::tuple keys = randomModule_->attr("split")(rngKey_);
  if (keys.size() != 2) {
    throw std::runtime_error(absl::StrFormat("Tried to split key, but got back %d things", keys.size()));
  }
  rngKey_ = keys[0];
  return keys[1];
}

py::object JaxInterface::convertToNumpy(const Observation &observation) {
  py::array_t<float> array(2);
  auto mutableArray = array.mutable_unchecked<1>();
  mutableArray(0) = observation.ourHp_ / 2660.0f;
  mutableArray(1) = observation.opponentHp_ / 2660.0f;
  return array;
}

} // namespace rl