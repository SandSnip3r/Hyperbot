#ifndef RL_JAX_INTERFACE_HPP_
#define RL_JAX_INTERFACE_HPP_

#include "rl/observation.hpp"

#include <pybind11/pybind11.h>

#include <optional>

namespace rl {

class JaxInterface {
public:
  JaxInterface() = default;
  ~JaxInterface();
  void initialize();
  void train();
  int selectAction(const Observation &observation);
private:
  static constexpr int kSeed{0};
  std::optional<pybind11::module> jaxModule_;
  std::optional<pybind11::module> randomModule_;
  std::optional<pybind11::object> rngKey_;
  std::optional<pybind11::object> nnxRngs_;
  std::optional<pybind11::object> modelGraph_;
  std::optional<pybind11::object> modelWeights_;

  pybind11::object getNextRngKey();
  pybind11::object convertToNumpy(const Observation &observation);
};

// We need 3 main interfaces:
// 1. Get an action choice from the latest model.
// 2. Get an action choice from older static model.
// 3. Train the latest model.

} // namespace rl

#endif // RL_JAX_INTERFACE_HPP_