#include "shared/silkroad_security.h"
#include "packetEnums.hpp"

#include <string>

#ifndef PACKET_PARSING_HPP
#define PACKET_PARSING_HPP

namespace PacketParsing {

class PacketParser {
protected:
  bool parsed_{false};
  void parsedCheck();
  virtual void parsePacket() = 0;
};

PacketParser* newPacketParser(const PacketContainer &packet);


class ClientChatPacket : public PacketParser {
private:
  const PacketContainer &packet_;
  void parsePacket() override;
public:
  ClientChatPacket(const PacketContainer &packet);
  PacketEnums::ChatType chatType();
  uint8_t chatIndex();
  const std::string& receiverName();
  const std::string& message();
private:
  PacketEnums::ChatType chatType_;
  uint8_t chatIndex_;
  std::string receiverName_;
  std::string message_;
};

// class ServerCharacterPacket : public PacketParser {
// private:
//   const PacketContainer &packet_;
//   int maxHp_;
//   void parsePacket() override;
// public:
//   ServerCharacterPacket(const PacketContainer &packet);
//   int maxHp();
// };

// class ServerHpMpUpdatePacket : public PacketParser {
// private:
//   const PacketContainer &packet_;
//   int hp_;
//   void parsePacket() override;
// public:
//   ServerHpMpUpdatePacket(const PacketContainer &packet);
//   int hp();
// };

} // namespace PacketParsing

#endif // PACKET_PARSING_HPP