#include "hyperbot.hpp"

#include <ui_proto/request.pb.h>

#include <absl/log/log.h>
#include <absl/strings/str_format.h>

Hyperbot::~Hyperbot() {
  if (connectionThread_.joinable()) {
    connectionThread_.join();
  }
}

void Hyperbot::tryConnectAsync(std::string_view ipAddress, int32_t port) {
  LOG(INFO) << absl::StrFormat("Connecting to Hyperbot at %s:%d.", ipAddress, port);

  socket_ = zmq::socket_t(context_, zmq::socket_type::req);
  socket_.connect(absl::StrFormat("tcp://%s:%d", ipAddress, port));

  if (connectionThread_.joinable()) {
    VLOG(1) << "Joining existing connection thread";
    connectionThread_.join();
  }
  tryToConnect_ = true;
  connectionThread_ = std::thread([this]() {
    tryConnect();
  });
}

void Hyperbot::cancelConnect() {
  tryToConnect_ = false;
}

void Hyperbot::tryConnect() {
  // Do one send/recv to test the connection.
  proto::request::RequestMessage pingRequest;
  pingRequest.mutable_ping();
  std::string protoMsgAsStr;
  pingRequest.SerializeToString(&protoMsgAsStr);
  zmq::message_t zmqMsg;
  zmqMsg.rebuild(protoMsgAsStr.data(), protoMsgAsStr.size());
  std::optional<zmq::send_result_t> sendResult = socket_.send(zmqMsg, zmq::send_flags::none);

  if (!sendResult.has_value()) {
    LOG(WARNING) << "Failed to send message to Hyperbot.";
    connectionFailed();
    return;
  }

  LOG(INFO) << "Awaiting connection.";
  constexpr int kAttemptCount = 5;
  for (int i=0; i<kAttemptCount; ++i) {
    VLOG(1) << "Attempt " << i;
    if (!tryToConnect_) {
      connectionCancelled();
      return;
    }
    if (i > 0) {
      LOG(INFO) << "No reply from Hyperbot. Retrying... (attempt "  << i+1 << '/' << kAttemptCount << ").";
    }
    // Poll for a reply.
    std::chrono::milliseconds timeout(2000);
    std::vector<zmq::pollitem_t> items = { { socket_, 0, ZMQ_POLLIN, 0 } };
    const int pollResult = zmq::poll(items, timeout);
    if (pollResult == 1) {
      // Successfully received a reply.
      break;
    }
    if (i < kAttemptCount - 1) {
      // Retry.
      continue;
    }

    LOG(WARNING) << "No reply from Hyperbot. Giving up.";
    connectionFailed();
    return;
  }


  // Receive the reply.
  zmq::message_t reply;
  std::optional<zmq::recv_result_t> receiveResult = socket_.recv(reply, zmq::recv_flags::none);
  if (!receiveResult.has_value()) {
    LOG(WARNING) << "Failed to receive reply from Hyperbot.";
    connectionFailed();
    return;
  }
  VLOG(1) << "Received response " << reply.to_string();
  connected();
}
