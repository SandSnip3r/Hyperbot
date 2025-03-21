#include "hyperbot.hpp"

#include <absl/log/log.h>
#include <absl/strings/str_format.h>

using namespace proto;

Hyperbot::~Hyperbot() {
  if (connectionThread_.joinable()) {
    connectionThread_.join();
  }
}

void Hyperbot::tryConnectAsync(std::string_view ipAddress, int32_t port) {
  ipAddress_ = ipAddress;
  LOG(INFO) << absl::StrFormat("Connecting to Hyperbot at %s:%d.", ipAddress_, port);

  socket_ = zmq::socket_t(context_, zmq::socket_type::req);
  socket_.connect(absl::StrFormat("tcp://%s:%d", ipAddress_, port));

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
  doAction(rl_ui_messages::DoAction::kStartTraining);
}

void Hyperbot::stopTraining() {
  doAction(rl_ui_messages::DoAction::kStopTraining);
}

void Hyperbot::requestCheckpointList() {
  using namespace proto;
  rl_ui_messages::RequestMessage request;
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

  rl_ui_messages::ReplyMessage replyMsg;
  bool receiveSuccess = replyMsg.ParseFromArray(reply.data(), reply.size());
  if (!receiveSuccess) {
    LOG(WARNING) << "Failed to parse reply";
    return;
  }
  VLOG(1) << "Successfully parsed reply: " << replyMsg.DebugString();
  if (replyMsg.body_case() != rl_ui_messages::ReplyMessage::BodyCase::kCheckpointList) {
    LOG(WARNING) << "Received unexpected reply";
    return;
  }
  rl_ui_messages::CheckpointList checkpointList = replyMsg.checkpoint_list();
  QStringList checkpointListStr;
  for (const rl_ui_messages::Checkpoint &checkpoint : checkpointList.checkpoints()) {
    checkpointListStr.append(QString::fromStdString(checkpoint.name()));
  }
  checkpointListReceived(checkpointListStr);
}

void Hyperbot::tryConnect() {
  const std::chrono::milliseconds kReplyTimeout(2000);

  // Do one send/recv to test the connection.
  rl_ui_messages::RequestMessage pingRequest;
  pingRequest.mutable_ping();
  bool sendSuccess = sendMessage(pingRequest);
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
    std::vector<zmq::pollitem_t> items = { { socket_, 0, ZMQ_POLLIN, 0 } };
    const int pollResult = zmq::poll(items, kReplyTimeout);
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
  zmq::recv_result_t receiveResult = socket_.recv(reply, zmq::recv_flags::none);
  if (!receiveResult) {
    LOG(WARNING) << "Failed to receive ping reply from Hyperbot.";
    connectionFailed();
    return;
  }
  VLOG(2) << "Received ping reply data \"" << reply.to_string() << "\"";

  // Ensure the reply is the message we expected.
  rl_ui_messages::ReplyMessage pingReplyMsg;
  bool parseSuccess = pingReplyMsg.ParseFromArray(reply.data(), reply.size());
  if (!parseSuccess) {
    LOG(WARNING) << "Failed to parse ping reply";
    connectionFailed();
    return;
  }
  VLOG(2) << "Received ping reply \"" << pingReplyMsg.DebugString() << "\"";

  if (pingReplyMsg.body_case() != rl_ui_messages::ReplyMessage::BodyCase::kPingAck) {
    LOG(WARNING) << "Received unexpected reply";
    connectionFailed();
    return;
  }

  // Now that we know we are connected to Hyperbot, request the port of its broadcast socket, so that we can connect to that too.
  rl_ui_messages::RequestMessage broadcastPortRequest;
  broadcastPortRequest.mutable_request_broadcast_port();
  sendSuccess = sendMessage(broadcastPortRequest);
  if (!sendSuccess) {
    connectionFailed();
    return;
  }

  // Again, poll for a reply, in case Hyperbot went down after we successfully connected.
  std::vector<zmq::pollitem_t> items = { { socket_, 0, ZMQ_POLLIN, 0 } };
  const int pollResult = zmq::poll(items, kReplyTimeout);
  if (pollResult != 1) {
    LOG(WARNING) << "No reply from Hyperbot when getting broadcast port.";
    connectionFailed();
    return;
  }

  // Receive the reply.
  receiveResult = socket_.recv(reply, zmq::recv_flags::none);
  if (!receiveResult) {
    LOG(WARNING) << "Failed to receive broadcast port reply from Hyperbot.";
    connectionFailed();
    return;
  }

  VLOG(1) << "Received broadcast port reply data \"" << reply.to_string() << "\"";
  rl_ui_messages::ReplyMessage broadcastPortReplyMsg;
  parseSuccess = broadcastPortReplyMsg.ParseFromArray(reply.data(), reply.size());
  if (!parseSuccess) {
    LOG(WARNING) << "Failed to parse reply";
    return;
  }
  VLOG(2) << "Received broadcast reply \"" << broadcastPortReplyMsg.DebugString() << "\"";

  if (broadcastPortReplyMsg.body_case() != rl_ui_messages::ReplyMessage::BodyCase::kBroadcastPort) {
    LOG(WARNING) << "Received unexpected reply";
    connectionFailed();
    return;
  }

  int32_t broadcastPort = broadcastPortReplyMsg.broadcast_port();
  LOG(INFO) << "Received broadcast port: " << broadcastPort;
  subscriber_ = zmq::socket_t(context_, zmq::socket_type::sub);
  subscriber_.set(zmq::sockopt::subscribe, "");
  subscriber_.connect(absl::StrFormat("tcp://%s:%d", ipAddress_, broadcastPort));
  subscriberThread_ = std::thread([this]() {
    subscriberThreadFunc();
  });
  connected();
}

void Hyperbot::doAction(rl_ui_messages::DoAction::Action action) {
  LOG(INFO) << "Going to send DoAction-" << rl_ui_messages::DoAction::Action_Name(action) << " message.";
  rl_ui_messages::RequestMessage startTrainingRequest;
  rl_ui_messages::DoAction *doAction = startTrainingRequest.mutable_do_action();
  doAction->set_action(action);

  std::string protoMsgAsStr;
  startTrainingRequest.SerializeToString(&protoMsgAsStr);
  zmq::message_t zmqMsg(protoMsgAsStr);
  zmq::send_result_t sendResult = socket_.send(zmqMsg, zmq::send_flags::none);
  if (!sendResult) {
    LOG(WARNING) << "Failed to send message to Hyperbot.";
  }
  zmq::message_t reply;
  zmq::recv_result_t receiveResult = socket_.recv(reply, zmq::recv_flags::none);
  if (!receiveResult) {
    LOG(WARNING) << "Failed to receive reply";
    return;
  }

  rl_ui_messages::ReplyMessage replyMsg;
  bool parseSuccess = replyMsg.ParseFromArray(reply.data(), reply.size());
  if (!parseSuccess) {
    LOG(WARNING) << "Failed to parse reply";
    return;
  }
  VLOG(1) << "Successfully parsed reply: " << replyMsg.DebugString();
  if (replyMsg.body_case() != rl_ui_messages::ReplyMessage::BodyCase::kDoActionAck) {
    LOG(WARNING) << "Received unexpected reply";
    return;
  }
}

bool Hyperbot::sendMessage(const rl_ui_messages::RequestMessage &message) {
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


void Hyperbot::subscriberThreadFunc() {
  while (true) {
    zmq::message_t message;
    subscriber_.recv(message);
    rl_ui_messages::BroadcastMessage broadcastMessage;
    broadcastMessage.ParseFromArray(message.data(), message.size());
    LOG(INFO) << "Received broadcast message \"" << broadcastMessage.DebugString() << "\"";
  }
}