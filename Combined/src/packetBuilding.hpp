#include "opcode.hpp"
#include "packetEnums.hpp"
#include "shared/silkroad_security.h"

#ifndef PACKET_BUILDING_HPP
#define PACKET_BUILDING_HPP

namespace PacketBuilding {

class PacketBuilder {
protected:
  Opcode opcode_;
  PacketContainer packet(const StreamUtility &stream, bool encrypted, bool massive) const;
  PacketBuilder(Opcode opcode);
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