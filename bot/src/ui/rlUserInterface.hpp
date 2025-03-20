#ifndef UI_RL_USERINTERFACE_HPP_
#define UI_RL_USERINTERFACE_HPP_

#include <zmq.hpp>

#include <atomic>
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
  zmq::context_t &context_;
  broker::EventBroker &eventBroker_;
  std::atomic<bool> keepRunning_;
  zmq::socket_t publisher_{context_, zmq::socket_type::pub};
  std::thread thr_;

  void run();
  zmq::message_t handleRequest(const zmq::message_t &request);
};

} // namespace ui

#endif // UI_RL_USERINTERFACE_HPP_