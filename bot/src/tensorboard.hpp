#ifndef TENSORBOARD_HPP_
#define TENSORBOARD_HPP_

#include <pybind11/pybind11.h>

#include <mutex>
#include <optional>
#include <string_view>

class Tensorboard {
public:
  static Tensorboard& instance();

  void addScalar(std::string_view name, double yValue, double xValue);
private:
  Tensorboard();
  ~Tensorboard();

  std::mutex mutex_;
  std::optional<pybind11::object> summaryWriter_;
};

#endif // TENSORBOARD_HPP_