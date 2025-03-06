#ifndef RL_JAX_INTERFACE_HPP_
#define RL_JAX_INTERFACE_HPP_

#include <pybind11/pybind11.h>

namespace rl {

class JaxInterface {
public:
  JaxInterface();
  void train();
private:
  pybind11::module jaxModule_;
};

} // namespace rl

#endif // RL_JAX_INTERFACE_HPP_