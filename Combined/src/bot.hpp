#include "brokerSystem.hpp"
#include "shared/silkroad_security.h"
#include <functional>

#ifndef BOT_HPP_
#define BOT_HPP_

class Bot {
  //This contains all intelligence that acts upon packets
public:
  Bot(BrokerSystem &broker);
  // void configure(Config &config);
  // bool packetReceived(const PacketContainer &packet, PacketContainer::Direction packetDirection);
private:
  void commandHandler(const std::string &command);
  bool handleClientChat(std::unique_ptr<PacketParsing::PacketParser> &packetParser);
  BrokerSystem &broker_;
};

#endif