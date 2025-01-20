#include "hyperbot.hpp"

#include <absl/log/log.h>
#include <absl/strings/str_format.h>

bool Hyperbot::connect(std::string_view ipAddress, int32_t port) {
  LOG(INFO) << "Connecting to bot at " << ipAddress << ":" << port;
  socket_.connect(absl::StrFormat("tcp://%s:%d", ipAddress, port));
  // Do one send/recv to test the connection.
  std::optional<zmq::send_result_t> sendResult = socket_.send(zmq::str_buffer("Hello"), zmq::send_flags::none);
  if (!sendResult.has_value()) {
    LOG(WARNING) << "Failed to send message to bot";
    return false;
  }

  // Receive the reply.
  zmq::message_t reply;
  std::optional<zmq::recv_result_t> receiveResult = socket_.recv(reply, zmq::recv_flags::none);
  if (!receiveResult.has_value()) {
    LOG(WARNING) << "Failed to receive reply from bot";
    return false;
  }
  LOG(INFO) << "Received response " << reply.to_string();
  return true;
}
