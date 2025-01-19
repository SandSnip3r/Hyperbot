#include "hyperbot.hpp"

#include <absl/log/log.h>

void Hyperbot::connect(std::string_view ipAddress, int32_t port) {
  LOG(INFO) << "Connecting to bot at " << ipAddress << ":" << port;
}
