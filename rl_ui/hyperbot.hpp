#ifndef HYPERBOT_HPP_
#define HYPERBOT_HPP_

#include <zmq.hpp>

#include <cstdint>
#include <string_view>

class Hyperbot {
public:
  // Tries to connect to the Hyperbot server. Returns true if successful.
  bool connect(std::string_view ipAddress, int32_t port);
private:
  zmq::context_t context_;
  zmq::socket_t socket_{context_, zmq::socket_type::req};
};

#endif // HYPERBOT_HPP_
