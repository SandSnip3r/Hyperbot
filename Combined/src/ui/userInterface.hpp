#ifndef UI_USERINTERFACE_HPP_
#define UI_USERINTERFACE_HPP_

#include "broker/eventBroker.hpp"

#include "ui-proto/broadcast.pb.h"

#include <zmq.hpp>

namespace ui {

class UserInterface {
public:
  UserInterface(broker::EventBroker &eventBroker);
  ~UserInterface();
  void run();
  void broadcast(const broadcast::BroadcastMessage &broadcastProto);
private:
  zmq::context_t context_;
  zmq::socket_t publisher_{context_, zmq::socket_type::pub};
  broker::EventBroker &eventBroker_;
  std::thread thr_;
  void privateRun();
  void handle(const zmq::message_t &request);
};

} // namespace ui

#endif // UI_USERINTERFACE_HPP_