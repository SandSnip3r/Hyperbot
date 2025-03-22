#include "hyperbotConnectWorker.hpp"

#include <ui_proto/rl_ui_messages.pb.h>

#include <absl/log/log.h>

using namespace proto;

HyperbotConnectWorker::HyperbotConnectWorker(const std::string &ipAddress, int port, zmq::socket_t &socket, QObject *parent) : QObject(parent), ipAddress_(ipAddress), port_(port), socket_(socket) {}

HyperbotConnectWorker::~HyperbotConnectWorker() {}

void HyperbotConnectWorker::process() {
  auto sendMessage = [](const rl_ui_messages::RequestMessage &msg, zmq::socket_t &socket_) -> zmq::send_result_t {
    std::string protoMsgAsStr;
    msg.SerializeToString(&protoMsgAsStr);
    zmq::message_t zmqMsg(protoMsgAsStr);
    return socket_.send(zmqMsg, zmq::send_flags::none);
  };
  const std::chrono::milliseconds kReplyTimeout(500);

  // Send a ping request to test the connection.
  rl_ui_messages::RequestMessage pingRequest;
  pingRequest.mutable_ping();
  zmq::send_result_t sendResult = sendMessage(pingRequest, socket_);
  if (!sendResult) {
    emit connectionFailed();
    return;
  }

  LOG(INFO) << "Awaiting connection.";
  constexpr int kAttemptCount = 20;
  int attempt = 0;
  for (; attempt < kAttemptCount; ++attempt) {
    if (!tryToConnect_) {
      emit connectionCancelled();
      return;
    }
    if (attempt > 0) {
      LOG(INFO) << "No reply from Hyperbot. Retrying... (attempt "  << attempt+1 << '/' << kAttemptCount << ").";
    }
    VLOG(1) << "Attempt #" << attempt;
    std::vector<zmq::pollitem_t> items = { { socket_, 0, ZMQ_POLLIN, 0 } };
    int pollResult = zmq::poll(items, kReplyTimeout);
    if (pollResult == 1) {
      break;
    }
  }
  if (attempt == kAttemptCount) {
    LOG(WARNING) << "No reply from Hyperbot. Giving up.";
    emit connectionFailed();
    return;
  }

  zmq::message_t reply;
  zmq::recv_result_t receiveResult = socket_.recv(reply, zmq::recv_flags::none);
  if (!receiveResult) {
    emit connectionFailed();
    return;
  }
  rl_ui_messages::ReplyMessage pingReplyMsg;
  if (!pingReplyMsg.ParseFromArray(reply.data(), reply.size()) ||
    pingReplyMsg.body_case() != rl_ui_messages::ReplyMessage::kPingAck) {
    emit connectionFailed();
    return;
  }

  // Request the broadcast port.
  rl_ui_messages::RequestMessage broadcastPortRequest;
  broadcastPortRequest.mutable_request_broadcast_port();
  sendResult = sendMessage(broadcastPortRequest, socket_);
  if (!sendResult) {
    emit connectionFailed();
    return;
  }

  std::vector<zmq::pollitem_t> items = { { socket_, 0, ZMQ_POLLIN, 0 } };
  int pollResult = zmq::poll(items, kReplyTimeout);
  if (pollResult != 1) {
    emit connectionFailed();
    return;
  }

  receiveResult = socket_.recv(reply, zmq::recv_flags::none);
  if (!receiveResult) {
    emit connectionFailed();
    return;
  }
  rl_ui_messages::ReplyMessage broadcastPortReplyMsg;
  if (!broadcastPortReplyMsg.ParseFromArray(reply.data(), reply.size()) ||
    broadcastPortReplyMsg.body_case() != rl_ui_messages::ReplyMessage::kBroadcastPort) {
    emit connectionFailed();
    return;
  }

  int32_t broadcastPort = broadcastPortReplyMsg.broadcast_port();
  emit connected(broadcastPort);
}