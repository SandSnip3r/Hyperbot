#include "tensorboard.hpp"

#include "flags.hpp"

#include <absl/flags/flag.h>
#include <absl/log/log.h>

Tensorboard& Tensorboard::instance() {
  static Tensorboard tensorboard;
  return tensorboard;
}

Tensorboard::Tensorboard() {
  using namespace pybind11::literals;
  pybind11::gil_scoped_acquire acquire;
  pybind11::module tensorboardX = pybind11::module::import("tensorboardX");
  summaryWriter_ = tensorboardX.attr("SummaryWriter")("flush_secs"_a=1);
}

Tensorboard::~Tensorboard() {
  pybind11::gil_scoped_acquire acquire;
  if (summaryWriter_.has_value()) {
    summaryWriter_.reset();
  }
}

void Tensorboard::addScalar(std::string_view name, double yValue, double xValue) {
  std::unique_lock lock(mutex_);
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
  pybind11::gil_scoped_acquire acquire;
  summaryWriter_->attr("add_scalar")(name, yValue, xValue);
}