#include "brokerSystem.hpp"
#include "loginModule.hpp"
#include "shared/silkroad_security.h"
#include <functional>

#ifndef BOT_HPP_
#define BOT_HPP_

class Bot {
  //This contains all intelligence that acts upon packets
public:
  Bot(BrokerSystem &broker);
  // void configure(Config &config);
private:
  // LoginState loginState_{LoginState::kWaitingForServerList};
  BrokerSystem &broker_;
  LoginModule loginModule_{broker_};
  // bool handleClientChat(std::unique_ptr<PacketParsing::PacketParser> &packetParser);
  // void commandHandler(const std::string &command);
};

#endif