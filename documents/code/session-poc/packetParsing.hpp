#include "packet.hpp"

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

PacketParser* newPacketParser(const Packet &packet);

class ServerCharacterPacket : public PacketParser {
private:
  const Packet &packet_;
  int maxHp_;
  void parsePacket() override;
public:
  ServerCharacterPacket(const Packet &packet);
  int maxHp();
};

class ServerHpMpUpdatePacket : public PacketParser {
private:
  const Packet &packet_;
  int hp_;
  void parsePacket() override;
public:
  ServerHpMpUpdatePacket(const Packet &packet);
  int hp();
};

} // namespace PacketParsing

#endif // PACKET_PARSING_HPP