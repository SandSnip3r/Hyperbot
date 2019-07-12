#ifndef BOT_HPP
#define BOT_HPP

#include "event/include/eventBroker.hpp"
#include "packetBroker.hpp"

class Bot {
public:
  Bot(PacketBroker packetbroker);
  void run() {
    eventBroker_.run();
  }
private:
  event::EventBroker eventBroker_;
  PacketBroker packetbroker_;
};

#endif // BOT_HPP