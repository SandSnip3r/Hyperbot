#include "opcode.hpp"
#include "packetEnums.hpp"
#include "shared/silkroad_security.h"

#include <array>

#ifndef PACKET_BUILDING_HPP
#define PACKET_BUILDING_HPP

namespace PacketBuilding {

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
  PacketEnums::CharacterSelectionAction action_;
public:
  CharacterSelectionActionPacketBuilder(PacketEnums::CharacterSelectionAction action);
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
  PacketEnums::ChatType chatType_;
  uint32_t senderId_;
  std::string senderName_;
  std::string message_;
public:
  ServerChatPacketBuilder(PacketEnums::ChatType chatType, const std::string &message);
  ServerChatPacketBuilder(PacketEnums::ChatType chatType, uint32_t senderId, const std::string &message);
  ServerChatPacketBuilder(PacketEnums::ChatType chatType, const std::string &senderName, const std::string &message);
  PacketContainer packet() const;
};

} // namespace PacketBuilding

#endif // PACKET_BUILDING_HPP