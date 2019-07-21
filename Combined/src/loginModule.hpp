#ifndef LOGINMODULE_HPP_
#define LOGINMODULE_HPP_

#include "../../common/divisionInfo.hpp"
#include "brokerSystem.hpp"
#include "configData.hpp"
#include "packetParsing.hpp"
#include "shared/silkroad_security.h"

#include <array>
#include <memory>
#include <string>

class LoginModule {
private:
  // enum class LoginState {
  //   kWaitingForServerList,
  //   kWaitingForClientCafe,
  //   kWaitingForServerAuthInfo,
  //   kCafeSent,
  //   kLoginResponseReceived
  // };
public:
  LoginModule(const config::CharacterLoginData &loginData, const pk2::DivisionInfo &divisionInfo, BrokerSystem &brokerSystem);
  bool handlePacket(std::unique_ptr<PacketParsing::PacketParser> &packetParser);
private:
  const config::CharacterLoginData &loginData_;
  const pk2::DivisionInfo &divisionInfo_;
  BrokerSystem &broker_;
  bool loggingIn_ = false;
  uint32_t token_;
  uint16_t shardId_;
  const std::array<uint8_t,6> kMacAddress_ = {0,0,0,0,0,0};
  void cafeReceived();
  void serverListReceived(PacketParsing::LoginServerListPacket &packet);
  void loginResponseReceived(PacketParsing::LoginResponsePacket &packet);
  void loginClientInfoReceived(PacketParsing::LoginClientInfoPacket &packet);
  bool unknownPacketReceived(PacketParsing::UnknownPacket &packet);
  void serverAuthReceived(PacketParsing::ServerAuthResponsePacket &packet);
  void charListReceived(PacketParsing::ServerAgentCharacterSelectionActionResponsePacket &packet);
  void charSelectionJoinResponseReceived(PacketParsing::ServerAgentCharacterSelectionJoinResponsePacket &packet);
};

#endif // LOGINMODULE_HPP_