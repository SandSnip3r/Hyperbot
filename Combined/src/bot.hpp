#include "loginModule.hpp"
#include "shared/silkroad_security.h"
#include <functional>

#ifndef BOT_HPP_
#define BOT_HPP_

class Bot {
  //This contains all intelligence that acts upon packets
public:
  Bot(std::function<void(PacketContainer&, PacketContainer::Direction)> injectionFunction);
  // void configure(Config &config);
  bool packetReceived(const PacketContainer &packet, PacketContainer::Direction packetDirection);
private:
	std::function<void(PacketContainer&, PacketContainer::Direction)> injectionFunction_;
  LoginModule loginModule_;
};

#endif