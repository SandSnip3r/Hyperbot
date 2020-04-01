#include "../opcode.hpp"
#include "../enums/packetEnums.hpp"
#include "../../shared/silkroad_security.h"

#include <array>

#ifndef PACKET_BUILDING_HPP
#define PACKET_BUILDING_HPP

namespace packet::building {

namespace client_item_move {
PacketContainer withinInventory(uint8_t srcSlot, uint8_t destSlot, uint16_t quantity);
} // namespace client_item_move

class PacketBuilder {
protected:
  Opcode opcode_;
  PacketContainer packet(const StreamUtility &stream, bool encrypted, bool massive) const;
  PacketBuilder(Opcode opcode);
};

class ClientAgentSelectionJoinPacketBuilder : public PacketBuilder {
private:
  const std::string name_;
public:
  ClientAgentSelectionJoinPacketBuilder(const std::string &name);
  PacketContainer packet() const;
};

class CharacterSelectionActionPacketBuilder : public PacketBuilder {
private:
  packet::enums::CharacterSelectionAction action_;
public:
  CharacterSelectionActionPacketBuilder(packet::enums::CharacterSelectionAction action);
  PacketContainer packet() const;
};

class ClientAuthPacketBuilder : public PacketBuilder {
private:
  uint32_t loginToken_;
  std::string kUsername_;
  std::string kPassword_;
  uint8_t kLocale_;
  std::array<uint8_t,6> macAddress_;
public:
  ClientAuthPacketBuilder(uint32_t loginToken, const std::string &kUsername, const std::string &kPassword, uint8_t kLocale, const std::array<uint8_t,6> &macAddress);
  PacketContainer packet() const;
};

class ClientCaptchaBuilder : public PacketBuilder {
private:
  std::string answer_;
public:
  ClientCaptchaBuilder(const std::string &answer);
  PacketContainer packet() const;
};

class ClientUseItemBuilder : public PacketBuilder {
private:
  uint8_t slotNum_;
  uint16_t itemData_;
public:
  ClientUseItemBuilder(uint8_t slotNum, uint16_t itemData);
  PacketContainer packet() const;
};

class LoginAuthPacketBuilder : public PacketBuilder {
private:
  uint8_t locale_;
  const std::string username_;
  const std::string password_;
  uint16_t shardId_;
public:
  LoginAuthPacketBuilder(uint8_t locale, const std::string &username, const std::string &password, uint16_t shardId);
  PacketContainer packet() const;
};

class ClientMovementRequestPacketBuilder : public PacketBuilder {
private:
  bool hasDestination_;
  uint16_t regionId_;
  uint32_t xOffset_;
  uint32_t yOffset_;
  uint32_t zOffset_;
  uint16_t angle_;
public:
  ClientMovementRequestPacketBuilder(uint16_t angle);
  ClientMovementRequestPacketBuilder(uint16_t regionId, uint32_t xOffset, uint32_t yOffset, uint32_t zOffset);
  PacketContainer packet() const;
};

class ServerChatPacketBuilder : public PacketBuilder {
private:
  packet::enums::ChatType chatType_;
  uint32_t senderId_;
  std::string senderName_;
  std::string message_;
public:
  ServerChatPacketBuilder(packet::enums::ChatType chatType, const std::string &message);
  ServerChatPacketBuilder(packet::enums::ChatType chatType, uint32_t senderId, const std::string &message);
  ServerChatPacketBuilder(packet::enums::ChatType chatType, const std::string &senderName, const std::string &message);
  PacketContainer packet() const;
};

} // namespace packet::building

#endif // PACKET_BUILDING_HPP