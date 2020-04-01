#ifndef LOGIN_MODULE_HPP_
#define LOGIN_MODULE_HPP_

#include "../broker/packetBroker.hpp"
#include "../config/configData.hpp"
#include "../packet/parsing/packetParser.hpp"
#include "../packet/parsing/parsedPacket.hpp"
#include "../shared/silkroad_security.h"
#include "../../../common/pk2/divisionInfo.hpp"

#include <array>
#include <string>

namespace module {

class LoginModule {
public:
  // enum class LoginState {
  //   kWaitingForServerList,
  //   kWaitingForClientCafe,
  //   kWaitingForServerAuthInfo,
  //   kCafeSent,
  //   kLoginResponseReceived
  // };
  LoginModule(broker::PacketBroker &brokerSystem,
              const packet::parsing::PacketParser &packetParser,
              const config::CharacterLoginData &loginData,
              const pk2::DivisionInfo &divisionInfo);
  bool handlePacket(const PacketContainer &packet);
private:
  broker::PacketBroker &broker_;
  const packet::parsing::PacketParser &packetParser_;
  const config::CharacterLoginData &loginData_;
  const pk2::DivisionInfo &divisionInfo_;
  bool loggingIn_ = false;
  uint32_t token_;
  uint16_t shardId_;
  const std::array<uint8_t,6> kMacAddress_ = {0,0,0,0,0,0};
  const std::string kCaptchaAnswer_ = "";
  void serverListReceived(const packet::parsing::ParsedLoginServerList &packet);
  void loginResponseReceived(const packet::parsing::ParsedLoginResponse &packet);
  void loginClientInfoReceived(const packet::parsing::ParsedLoginClientInfo &packet);
  bool unknownPacketReceived(const packet::parsing::ParsedUnknown &packet);
  void serverAuthReceived(const packet::parsing::ParsedServerAuthResponse &packet);
  void charListReceived(const packet::parsing::ParsedServerAgentCharacterSelectionActionResponse &packet);
  void charSelectionJoinResponseReceived(const packet::parsing::ParsedServerAgentCharacterSelectionJoinResponse &packet);
};

} // namespace module

#endif // LOGINMODULE_HPP_