#ifndef BOT_HPP_
#define BOT_HPP_

#include "brokerSystem.hpp"
#include "configData.hpp"
#include "loginModule.hpp"
#include "shared/silkroad_security.h"
#include <functional>

class Bot {
  //This contains all intelligence that acts upon packets
public:
  Bot(const config::ConfigData &configData, BrokerSystem &broker);
  // void configure(Config &config);
private:
  // LoginState loginState_{LoginState::kWaitingForServerList};
  const config::ConfigData &configData_;
  BrokerSystem &broker_;
  LoginModule loginModule_{configData_, broker_};
  // bool handleClientChat(std::unique_ptr<PacketParsing::PacketParser> &packetParser);
  // void commandHandler(const std::string &command);
};

#endif