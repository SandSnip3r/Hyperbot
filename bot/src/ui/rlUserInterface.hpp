#ifndef UI_RL_USERINTERFACE_HPP_
#define UI_RL_USERINTERFACE_HPP_

#include <ui_proto/rl_ui_messages.pb.h>

#include <zmq.hpp>

#include <absl/strings/str_format.h>

#include <atomic>
#include <mutex>
#include <string>
#include <thread>

namespace broker {
class EventBroker;
} // namespace broker

namespace ui {

class RlUserInterface {
public:
  RlUserInterface(zmq::context_t &context, broker::EventBroker &eventBroker);
  ~RlUserInterface();
  void initialize();
  void runAsync();

  void sendCheckpointList(const std::vector<std::string> &checkpointList);
  void sendCheckpointAlreadyExists(const std::string &checkpointName);
private:
  static constexpr std::chrono::milliseconds kHeartbeatInterval{250};
  const std::string kReqReplyAddress{"tcp://*:5555"};
  static constexpr int kPublisherPort{5556};
  const std::string kPublisherAddress{absl::StrFormat("tcp://*:%d", kPublisherPort)};
  zmq::context_t &context_;
  broker::EventBroker &eventBroker_;
  std::atomic<bool> keepRunning_;
  std::mutex publisherMutex_;
  zmq::socket_t publisher_{context_, zmq::socket_type::pub};
  std::thread requestHandlingThread_;
  std::thread broadcastHeartbeatThread_;

  void requestLoop();
  void heartbeatLoop();
  void handleRequest(const zmq::message_t &request, zmq::socket_t &socket);
  void broadcastMessage(const proto::rl_ui_messages::BroadcastMessage &message);
};

} // namespace ui

#endif // UI_RL_USERINTERFACE_HPP_