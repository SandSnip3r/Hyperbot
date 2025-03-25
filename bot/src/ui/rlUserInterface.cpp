#include "broker/eventBroker.hpp"
#include "event/event.hpp"
#include "ui/rlUserInterface.hpp"

#include <absl/log/log.h>

using namespace proto;

namespace ui {

RlUserInterface::RlUserInterface(zmq::context_t &context, broker::EventBroker &eventBroker) : context_(context), eventBroker_(eventBroker) {

}

RlUserInterface::~RlUserInterface() {
  VLOG(1) << "Destructing UserInterface";
  if (requestHandlingThread_.joinable() || broadcastHeartbeatThread_.joinable()) {
    keepRunning_ = false;
    if (requestHandlingThread_.joinable()) {
      requestHandlingThread_.join();
    }
    if (broadcastHeartbeatThread_.joinable()) {
      broadcastHeartbeatThread_.join();
    }
  }

}

void RlUserInterface::initialize() {

}

void RlUserInterface::runAsync() {
  if (requestHandlingThread_.joinable() || broadcastHeartbeatThread_.joinable()) {
    throw std::runtime_error("UserInterface::runAsync called while already running");
  }
  // Set up publisher
  try {
    VLOG(1) << "RlUserInterface:publisher binding to " << kPublisherAddress;
    publisher_.bind(kPublisherAddress);

    // Run the request receiver in another thread
    keepRunning_ = true;
    requestHandlingThread_ = std::thread(&RlUserInterface::requestLoop, this);
    broadcastHeartbeatThread_ = std::thread(&RlUserInterface::heartbeatLoop, this);
  } catch (const std::exception &ex) {
    LOG(ERROR) << "Exception while binding to UI: \"" << ex.what() << "\"";
  } catch (...) {
    LOG(ERROR) << "Exception while binding to UI";
  }
}

// ================================================================================
// ===================================== Send =====================================
// ================================================================================

void RlUserInterface::sendCheckpointList(const std::vector<std::string> &checkpointList) {
  rl_ui_messages::BroadcastMessage msg;
  rl_ui_messages::CheckpointList *checkpointListMsg = msg.mutable_checkpoint_list();
  for (const std::string &checkpointName : checkpointList) {
    rl_ui_messages::Checkpoint *checkpointMsg = checkpointListMsg->add_checkpoints();
    checkpointMsg->set_name(checkpointName);
  }
  broadcastMessage(msg);
}

void RlUserInterface::sendCheckpointAlreadyExists(const std::string &checkpointName) {
  rl_ui_messages::BroadcastMessage msg;
  *msg.mutable_checkpoint_already_exists() = checkpointName;
  broadcastMessage(msg);
}

void RlUserInterface::sendSavingCheckpoint() {
  rl_ui_messages::BroadcastMessage msg;
  msg.mutable_saving_checkpoint();
  broadcastMessage(msg);
}

// ================================================================================
// ================================================================================
// ================================================================================

void RlUserInterface::requestLoop() {
  // Run request receiver
  zmq::socket_t socket(context_, zmq::socket_type::rep);
  VLOG(1) << "RlUserInterface:socket binding to " << kReqReplyAddress << "; this is the address which the UI should connect to.";
  socket.bind(kReqReplyAddress);
  while (keepRunning_) {
    // Wait for a request with a timeout so that we have an opportunity to check if we should stop running.
    zmq::message_t request;
    zmq::pollitem_t items[] = {{socket, 0, ZMQ_POLLIN, 0}};
    const std::chrono::duration timeout = std::chrono::milliseconds{100};
    zmq::poll(&items[0], 1, timeout);
    if ((items[0].revents & ZMQ_POLLIN) == 0) {
      // Did not receive anything yet.
      continue;
    }
    zmq::recv_result_t receiveResult = socket.recv(request, zmq::recv_flags::none);
    if (!receiveResult) {
      LOG(WARNING) << "Error receiving message";
      continue;
    }

    handleRequest(request, socket);
  }
}

void RlUserInterface::heartbeatLoop() {
  rl_ui_messages::BroadcastMessage msg;
  msg.mutable_heartbeat();
  while (keepRunning_) {
    broadcastMessage(msg);
    std::this_thread::sleep_for(kHeartbeatInterval);
  }
}

void RlUserInterface::handleRequest(const zmq::message_t &request, zmq::socket_t &socket) {
  rl_ui_messages::ReplyMessage replyMsg;
  // Parse the request
  rl_ui_messages::RequestMessage requestMsg;
  bool success = requestMsg.ParseFromArray(request.data(), request.size());
  if (!success) {
    throw std::runtime_error(absl::StrFormat("RlUserInterface received invalid data \"%s\"", request.str()));
  }
  LOG(INFO) << "Received request " << requestMsg.DebugString();
  switch (requestMsg.body_case()) {
    case rl_ui_messages::RequestMessage::BodyCase::kRequestBroadcastPort: {
      LOG(INFO) << "Received request for broadcast port";
      replyMsg.set_broadcast_port(kPublisherPort);
      break;
    }
    case rl_ui_messages::RequestMessage::BodyCase::kPing: {
      LOG(INFO) << "Received ping";
      replyMsg.mutable_ping_ack();
      break;
    }
    case rl_ui_messages::RequestMessage::BodyCase::kAsyncRequest: {
      const rl_ui_messages::AsyncRequest &asyncRequestMsg = requestMsg.async_request();
      if (asyncRequestMsg.body_case() == rl_ui_messages::AsyncRequest::BodyCase::kStartTraining) {
        eventBroker_.publishEvent(event::EventCode::kRlUiStartTraining);
      } else if (asyncRequestMsg.body_case() == rl_ui_messages::AsyncRequest::BodyCase::kStopTraining) {
        eventBroker_.publishEvent(event::EventCode::kRlUiStopTraining);
      } else if (asyncRequestMsg.body_case() == rl_ui_messages::AsyncRequest::BodyCase::kRequestCheckpointList) {
        eventBroker_.publishEvent(event::EventCode::kRlUiRequestCheckpointList);
      } else if (asyncRequestMsg.body_case() == rl_ui_messages::AsyncRequest::BodyCase::kSaveCheckpoint) {
        const rl_ui_messages::SaveCheckpoint &saveCheckpointMsg = asyncRequestMsg.save_checkpoint();
        eventBroker_.publishEvent<event::RlUiSaveCheckpoint>(saveCheckpointMsg.name());
      } else {
        throw std::runtime_error(absl::StrFormat("RlUserInterface received invalid async request \"%s\"", asyncRequestMsg.DebugString()));
      }
      replyMsg.mutable_async_request_ack();
      break;
    }
    default:
      throw std::runtime_error(absl::StrFormat("RlUserInterface received invalid message \"%s\"", requestMsg.DebugString()));
  }

  // Serialize to zmq message
  std::string protoMsgAsStr;
  replyMsg.SerializeToString(&protoMsgAsStr);
  // Immediately respond with the reply
  socket.send(zmq::message_t(protoMsgAsStr), zmq::send_flags::none);
}

void RlUserInterface::broadcastMessage(const rl_ui_messages::BroadcastMessage &message) {
  std::unique_lock lock(publisherMutex_);
  publisher_.send(zmq::message_t(message.SerializeAsString()), zmq::send_flags::none);
}

} // namespace ui