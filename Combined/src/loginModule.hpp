#ifndef LOGIN_MODULE_HPP_
#define LOGIN_MODULE_HPP_

#include "../../common/divisionInfo.hpp"
#include "brokerSystem.hpp"
#include "configData.hpp"
#include "packetParser.hpp"
#include "parsedPacket.hpp"
#include "shared/silkroad_security.h"

#include <array>
#include <string>

class LoginModule {
public:
  // enum class LoginState {
  //   kWaitingForServerList,
  //   kWaitingForClientCafe,
  //   kWaitingForServerAuthInfo,
  //   kCafeSent,
  //   kLoginResponseReceived
  // };
  LoginModule(BrokerSystem &brokerSystem,
              const packet::parsing::PacketParser &packetParser,
              const config::CharacterLoginData &loginData,
              const pk2::DivisionInfo &divisionInfo);
  bool handlePacket(const PacketContainer &packet);
private:
  BrokerSystem &broker_;
  const packet::parsing::PacketParser &packetParser_;
  const config::CharacterLoginData &loginData_;
  const pk2::DivisionInfo &divisionInfo_;
  bool loggingIn_ = false;
  uint32_t token_;
  uint16_t shardId_;
  const std::array<uint8_t,6> kMacAddress_ = {0,0,0,0,0,0};
  void serverListReceived(const packet::parsing::ParsedLoginServerList &packet);
  void loginResponseReceived(const packet::parsing::ParsedLoginResponse &packet);
  void loginClientInfoReceived(const packet::parsing::ParsedLoginClientInfo &packet);
  bool unknownPacketReceived(const packet::parsing::ParsedUnknown &packet);
  void serverAuthReceived(const packet::parsing::ParsedServerAuthResponse &packet);
  void charListReceived(const packet::parsing::ParsedServerAgentCharacterSelectionActionResponse &packet);
  void charSelectionJoinResponseReceived(const packet::parsing::ParsedServerAgentCharacterSelectionJoinResponse &packet);
};

#endif // LOGINMODULE_HPP_