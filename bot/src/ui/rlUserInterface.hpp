#ifndef UI_RL_USERINTERFACE_HPP_
#define UI_RL_USERINTERFACE_HPP_

#include <zmq.hpp>

#include <absl/strings/str_format.h>

#include <atomic>
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
private:
  const std::string kReqReplyAddress{"tcp://*:5555"};
  static constexpr int kPublisherPort{5556};
  const std::string kPublisherAddress{absl::StrFormat("tcp://*:%d", kPublisherPort)};
  zmq::context_t &context_;
  broker::EventBroker &eventBroker_;
  std::atomic<bool> keepRunning_;
  zmq::socket_t publisher_{context_, zmq::socket_type::pub};
  std::thread thr_;

  void run();
  void handleRequest(const zmq::message_t &request, zmq::socket_t &socket);
};

} // namespace ui

#endif // UI_RL_USERINTERFACE_HPP_