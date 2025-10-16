#include "packetParsing.hpp"

namespace PacketParsing {

PacketParser* newPacketParser(const Packet &packet) {
  // Given a packet's opcode, determine which parser is appropriate
  switch (packet.opcode()) {
    case Packet::Opcode::kServerCharacter:
      return new ServerCharacterPacket(packet);
    case Packet::Opcode::kServerHpMpUpdate:
      return new ServerHpMpUpdatePacket(packet);
  }
}

void PacketParser::parsedCheck() {
  // Lazy-eval mechanism
  if (!parsed_) {
    parsePacket();
    parsed_ = true;
  }
}

ServerCharacterPacket::ServerCharacterPacket(const Packet &packet) : packet_(packet) {}

void ServerCharacterPacket::parsePacket() {
  // Parse packet
  maxHp_ = packet_.maxHp();
}
int ServerCharacterPacket::maxHp() {
  parsedCheck();
  return maxHp_;
}

ServerHpMpUpdatePacket::ServerHpMpUpdatePacket(const Packet &packet) : packet_(packet) {}

void ServerHpMpUpdatePacket::parsePacket() {
  // Parse packet
  hp_ = packet_.hp();
}
int ServerHpMpUpdatePacket::hp() {
  parsedCheck();
  return hp_;
}

} // namespace PacketParsing