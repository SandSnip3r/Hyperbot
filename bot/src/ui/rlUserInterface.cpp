#include "broker/eventBroker.hpp"
#include "event/event.hpp"
#include "entity/self.hpp"
#include "ui/rlUserInterface.hpp"

// Tracy
#include <common/TracySystem.hpp>

#include <absl/log/log.h>

#include <ui_proto/rl_checkpointing.pb.h>

#include <chrono>

using namespace proto;

namespace ui {

RlUserInterface::RlUserInterface(zmq::context_t &context, broker::EventBroker &eventBroker)
    : context_(context), eventBroker_(eventBroker) {

}

RlUserInterface::~RlUserInterface() {
  VLOG(1) << "Destructing UserInterface";
  if (requestHandlingThread_.joinable() || broadcastHeartbeatThread_.joinable() || eventQueueThread_.joinable()) {
    keepRunning_ = false;
    if (requestHandlingThread_.joinable()) {
      requestHandlingThread_.join();
    }
    if (broadcastHeartbeatThread_.joinable()) {
      broadcastHeartbeatThread_.join();
    }
    if (eventQueueThread_.joinable()) {
      eventQueueThread_.join();
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
    eventQueueThread_ = std::thread(&RlUserInterface::eventQueueLoop, this);
  } catch (const std::exception &ex) {
    LOG(ERROR) << "Exception while binding to UI: \"" << ex.what() << "\"";
  } catch (...) {
    LOG(ERROR) << "Exception while binding to UI";
  }
}

// ================================================================================
// ===================================== Send =====================================
// ================================================================================

void RlUserInterface::sendCheckpointList(const std::vector<proto::rl_checkpointing::Checkpoint> &checkpointList) {
  VLOG(1) << "Sending checkpoint list to UI";
  rl_ui_messages::BroadcastMessage msg;
  rl_ui_messages::CheckpointList *checkpointListMsg = msg.mutable_checkpoint_list();
  for (const auto &checkpointProto : checkpointList) {
    rl_ui_messages::Checkpoint *checkpoint = checkpointListMsg->add_checkpoints();
    checkpoint->set_name(checkpointProto.checkpoint_name());
    checkpoint->set_timestamp_ms(checkpointProto.timestamp_ms());
    checkpoint->set_train_step_count(checkpointProto.step_count());
  }
  broadcastMessage(msg);
}

void RlUserInterface::sendCheckpointAlreadyExists(const std::string &checkpointName) {
  rl_ui_messages::BroadcastMessage msg;
  msg.set_checkpoint_already_exists(checkpointName);
  broadcastMessage(msg);
}

void RlUserInterface::sendSavingCheckpoint() {
  rl_ui_messages::BroadcastMessage msg;
  msg.mutable_saving_checkpoint();
  broadcastMessage(msg);
}

void RlUserInterface::sendCheckpointLoaded(const std::string &checkpointName) {
  rl_ui_messages::BroadcastMessage msg;
  rl_ui_messages::CheckpointLoaded *checkpointLoaded = msg.mutable_checkpoint_loaded();
  checkpointLoaded->set_name(checkpointName);
  broadcastMessage(msg);
}

void RlUserInterface::plot(std::string_view plotName, double x, double y) {
  rl_ui_messages::BroadcastMessage msg;
  rl_ui_messages::PlotData *plotData = msg.mutable_plot_data();
  plotData->set_name(std::string(plotName));
  plotData->set_x(x);
  plotData->set_y(y);
  broadcastMessage(msg);
}

// ================================================================================
// ================================================================================
// ================================================================================

void RlUserInterface::requestLoop() {
  tracy::SetThreadName("RlUserInterface::RequestLoop");
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
  tracy::SetThreadName("RlUserInterface::HeartbeatLoop");
  rl_ui_messages::BroadcastMessage msg;
  msg.mutable_heartbeat();
  while (keepRunning_) {
    broadcastMessage(msg);
    std::this_thread::sleep_for(kHeartbeatInterval);
  }
}

void RlUserInterface::eventQueueLoop() {
  tracy::SetThreadName("RlUserInterface::EventQueueLoop");
  const auto startTime = std::chrono::steady_clock::now();
  while (keepRunning_) {
    const auto now = std::chrono::steady_clock::now();
    double x = std::chrono::duration_cast<std::chrono::duration<double>>(now - startTime).count();
    double y = static_cast<double>(eventBroker_.queuedEventCount());
    plot("event_queue_size", x, y);
    std::this_thread::sleep_for(kEventQueueInterval);
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
      } else if (asyncRequestMsg.body_case() == rl_ui_messages::AsyncRequest::BodyCase::kLoadCheckpoint) {
        const rl_ui_messages::LoadCheckpoint &loadCheckpointMsg = asyncRequestMsg.load_checkpoint();
        eventBroker_.publishEvent<event::RlUiLoadCheckpoint>(loadCheckpointMsg.name());
      } else if (asyncRequestMsg.body_case() == rl_ui_messages::AsyncRequest::BodyCase::kDeleteCheckpoints) {
        const rl_ui_messages::DeleteCheckpoints &deleteCheckpointsMsg = asyncRequestMsg.delete_checkpoints();
        std::vector<std::string> checkpointNames;
        for (const std::string &checkpointName : deleteCheckpointsMsg.names()) {
          checkpointNames.push_back(checkpointName);
        }
        eventBroker_.publishEvent<event::RlUiDeleteCheckpoints>(checkpointNames);
      } else if (asyncRequestMsg.body_case() == rl_ui_messages::AsyncRequest::BodyCase::kRequestCharacterStatuses) {
        eventBroker_.publishEvent(event::EventCode::kRlUiRequestCharacterStatuses);
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

void RlUserInterface::sendCharacterStatus(const entity::Self &self) {
  rl_ui_messages::BroadcastMessage msg;
  auto *status = msg.mutable_character_status();
  status->set_name(self.name);
  status->set_current_hp(self.currentHp());
  status->set_max_hp(self.maxHp().value_or(self.currentHp()));
  status->set_current_mp(self.currentMp());
  status->set_max_mp(self.maxMp().value_or(self.currentMp()));
  broadcastMessage(msg);
}

void RlUserInterface::sendActiveStateMachine(const entity::Self &self, const std::string &stateMachine) {
  rl_ui_messages::BroadcastMessage msg;
  auto *payload = msg.mutable_active_state_machine();
  payload->set_name(self.name);
  payload->set_state_machine(stateMachine);
  broadcastMessage(msg);
}

void RlUserInterface::sendSkillCooldowns(const entity::Self &self) {
  rl_ui_messages::BroadcastMessage msg;
  auto *payload = msg.mutable_skill_cooldowns();
  payload->set_name(self.name);
  const auto &cooldownMap = self.skillEngine.getSkillCooldownEventIdMap();
  for (const auto &pair : cooldownMap) {
    std::optional<std::chrono::milliseconds> remaining =
        eventBroker_.timeRemainingOnDelayedEvent(pair.second);
    if (!remaining) {
      continue;
    }
    rl_ui_messages::SkillCooldown *cd = payload->add_cooldowns();
    cd->set_skill_id(pair.first);
    cd->set_remaining_ms(static_cast<int32_t>(remaining->count()));
    int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::system_clock::now().time_since_epoch())
                     .count();
    cd->set_timestamp_ms(now);
  }
  broadcastMessage(msg);
}

void RlUserInterface::sendQValues(const entity::Self &self, const std::vector<float> &qValues) {
  rl_ui_messages::BroadcastMessage msg;
  auto *payload = msg.mutable_character_q_values();
  payload->set_name(self.name);
  for (float q : qValues) {
    payload->add_q_values(q);
  }
  broadcastMessage(msg);
}

} // namespace ui