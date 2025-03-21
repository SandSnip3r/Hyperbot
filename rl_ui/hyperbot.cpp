#include "hyperbot.hpp"

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

void Hyperbot::startTraining() {
  doAction(proto::rl_ui_request::DoAction::kStartTraining);
}

void Hyperbot::stopTraining() {
  doAction(proto::rl_ui_request::DoAction::kStopTraining);
}

void Hyperbot::requestCheckpointList() {
  using namespace proto;
  rl_ui_request::RequestMessage request;
  request.mutable_request_checkpoint_list();
  const bool sendSuccess = sendMessage(request);
  if (!sendSuccess) {
    return;
  }
  VLOG(1) << "Successfully sent request for checkpoint list";

  zmq::message_t reply;
  zmq::recv_result_t receiveResult = socket_.recv(reply, zmq::recv_flags::none);
  if (!receiveResult) {
    LOG(WARNING) << "Failed to receive reply";
    return;
  }

  rl_ui_request::ResponseMessage responseMsg;
  bool receiveSuccess = responseMsg.ParseFromArray(reply.data(), reply.size());
  if (!receiveSuccess) {
    LOG(WARNING) << "Failed to parse reply";
    return;
  }
  VLOG(1) << "Successfully parsed response: " << responseMsg.DebugString();
  if (responseMsg.body_case() != rl_ui_request::ResponseMessage::BodyCase::kCheckpointList) {
    LOG(WARNING) << "Received unexpected response";
    return;
  }
  rl_ui_request::CheckpointList checkpointList = responseMsg.checkpoint_list();
  QStringList checkpointListStr;
  for (const rl_ui_request::Checkpoint &checkpoint : checkpointList.checkpoints()) {
    checkpointListStr.append(QString::fromStdString(checkpoint.name()));
  }
  checkpointListReceived(checkpointListStr);
}

void Hyperbot::tryConnect() {
  // Do one send/recv to test the connection.
  proto::rl_ui_request::RequestMessage pingRequest;
  pingRequest.mutable_ping();
  const bool sendSuccess = sendMessage(pingRequest);
  if (!sendSuccess) {
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

void Hyperbot::doAction(proto::rl_ui_request::DoAction::Action action) {
  LOG(INFO) << "Going to send DoAction-" << proto::rl_ui_request::DoAction::Action_Name(action) << " message.";
  proto::rl_ui_request::RequestMessage startTrainingRequest;
  proto::rl_ui_request::DoAction *doAction = startTrainingRequest.mutable_do_action();
  doAction->set_action(action);

  std::string protoMsgAsStr;
  startTrainingRequest.SerializeToString(&protoMsgAsStr);
  zmq::message_t zmqMsg(protoMsgAsStr);
  zmq::send_result_t sendResult = socket_.send(zmqMsg, zmq::send_flags::none);
  if (!sendResult) {
    LOG(WARNING) << "Failed to send message to Hyperbot.";
  }
}

bool Hyperbot::sendMessage(const proto::rl_ui_request::RequestMessage &message) {
  std::string protoMsgAsStr;
  message.SerializeToString(&protoMsgAsStr);
  zmq::message_t zmqMsg(protoMsgAsStr);
  std::optional<zmq::send_result_t> sendResult = socket_.send(zmqMsg, zmq::send_flags::none);

  if (!sendResult.has_value()) {
    LOG(WARNING) << "Failed to send message to Hyperbot.";
    return false;
  }
  return true;
}
