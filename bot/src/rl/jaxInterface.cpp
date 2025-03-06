#include "rl/jaxInterface.hpp"

#include <absl/log/log.h>

#include <thread>

namespace rl {

JaxInterface::JaxInterface() {
  jaxModule_ = pybind11::module::import("rl.python.myPython");
}

void JaxInterface::train() {
  // LOG(INFO) << "All aboard the JAX train!";
  jaxModule_.attr("func")(12345);
  std::this_thread::sleep_for(std::chrono::milliseconds(5000));
}

} // namespace rl