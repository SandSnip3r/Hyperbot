#include "../shared/silkroad_security.h"

#include <functional>
#include <string>

#ifndef CHATEVENTMODULE_HPP_
#define CHATEVENTMODULE_HPP_

namespace module {

class ChatEventModule {
public:
  ChatEventModule(std::function<void(PacketContainer&, PacketContainer::Direction)> injectionFunction);
  void serverChatReceived(const PacketContainer &packet);
  // void cafeSent();
  // void serverAuthInfoReceived(const PacketContainer &packet);
  // bool loginClientInfo(const PacketContainer &packet);
  // void serverLoginResultReceived(const PacketContainer &packet);
  // void serverCharacterListReceived(const PacketContainer &packet);
private:
  enum class ChatType { kAll=1, kNotice=7 };
	std::function<void(PacketContainer&, PacketContainer::Direction)> injectionFunction_;
  void writeAllChatMessage(const std::string &message);
  int chatIndex{0};
  int attempts{0};
  //
  // bool serverListParsed_ = false;
  // uint16_t shardId_;
  // bool gatewayLoginSuccessful_ = false;
  // uint32_t loginToken_;
  // bool loginPacketSentToGateway_ = false;
  // bool loginPacketSentToAgent_ = false;
  // bool requestedCharacterList_ = false;
  // const uint8_t kLocale_{0x16};
  // const std::string kUsername_{"sarkhan13"};
  // const std::string kPassword_{"2587597"};
  // const std::string kCharName_{"MadVillain"};
  // uint32_t macAddress_ = 0x24453945;
};

} // namespace module

#endif