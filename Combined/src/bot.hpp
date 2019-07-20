#ifndef BOT_HPP_
#define BOT_HPP_

#include "brokerSystem.hpp"
#include "configData.hpp"
#include "itemData.hpp"
#include "loginModule.hpp"
#include "shared/silkroad_security.h"
#include "skillData.hpp"

#include <functional>

class Bot {
  //This contains all intelligence that acts upon packets
public:
  Bot(const config::CharacterLoginData &loginData,
      const pk2::media::ItemData &itemData,
      const pk2::media::SkillData &skillData,
      BrokerSystem &broker);
  // void configure(Config &config);
private:
  // LoginState loginState_{LoginState::kWaitingForServerList};
  const config::CharacterLoginData &loginData_;
  const pk2::media::ItemData &itemData_;
  const pk2::media::SkillData &skillData_;
  BrokerSystem &broker_;
  LoginModule loginModule_{loginData_, broker_};
  // bool handleClientChat(std::unique_ptr<PacketParsing::PacketParser> &packetParser);
  // void commandHandler(const std::string &command);
};

#endif