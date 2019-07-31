#ifndef BOT_HPP_
#define BOT_HPP_

#include "brokerSystem.hpp"
#include "characterInfoModule.hpp"
#include "characterLoginData.hpp"
#include "gameData.hpp"
#include "loginModule.hpp"
#include "packetParser.hpp"

class Bot {
public:
  Bot(const config::CharacterLoginData &loginData,
      const pk2::media::GameData &gameData,
      BrokerSystem &broker);
private:
  const config::CharacterLoginData &loginData_;
  const pk2::media::GameData &gameData_;
  BrokerSystem &broker_;
  packet::parsing::PacketParser packetParser_{gameData_};
  CharacterInfoModule characterInfoModule_{broker_, packetParser_};
  LoginModule loginModule_{broker_, packetParser_, loginData_, gameData_.divisionInfo()};
};

#endif