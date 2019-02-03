#include "shared/silkroad_security.h"
#include <functional>

#ifndef LOGINMODULE_HPP_
#define LOGINMODULE_HPP_

class LoginModule {
public:
  LoginModule(std::function<void(PacketContainer&, PacketContainer::Direction)> injectionFunction);
  void serverListReceived(const PacketContainer &packet);
  void serverAuthInfoReceived(const PacketContainer &packet);
  bool loginClientInfo(const PacketContainer &packet);
private:
	std::function<void(PacketContainer&, PacketContainer::Direction)> injectionFunction_;
  uint16_t shardId_;
  bool gatewayLoginSuccessful_ = false;
  uint32_t loginToken_;
  bool loginPacketSent_ = false;
  const uint8_t kLocale_{0x16};
  const std::string kUsername_{"test"};
  const std::string kPassword_{"test"};
  uint32_t macAddress_ = 1000;
};

#endif