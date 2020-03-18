#include "opcode.hpp"
#include "packetParser.hpp"

#include <iostream>

namespace packet::parsing {

PacketParser::PacketParser(const pk2::media::GameData &gameData) :
      gameData_(gameData) {
  //
}

std::unique_ptr<ParsedPacket> PacketParser::parsePacket(const PacketContainer &packet) const {
  // Given a packet's opcode, determine which parsed packet type is appropriate
  switch (static_cast<Opcode>(packet.opcode)) {
    case Opcode::CLIENT_CHAT:
      return std::make_unique<ParsedClientChat>(packet);
    case Opcode::LOGIN_SERVER_LIST:
      return std::make_unique<ParsedLoginServerList>(packet);
    case Opcode::LOGIN_SERVER_AUTH_INFO:
      return std::make_unique<ParsedLoginResponse>(packet);
    case Opcode::LOGIN_CLIENT_INFO:
      return std::make_unique<ParsedLoginClientInfo>(packet);
    case Opcode::SERVER_LOGIN_RESULT:
      return std::make_unique<ParsedServerAuthResponse>(packet);
    case Opcode::SERVER_CHARACTER:
      return std::make_unique<ParsedServerAgentCharacterSelectionActionResponse>(packet);
    case Opcode::SERVER_INGAME_ACCEPT:
      return std::make_unique<ParsedServerAgentCharacterSelectionJoinResponse>(packet);
    case Opcode::SERVER_CHARDATA:
      return std::make_unique<ParsedServerAgentCharacterData>(packet, gameData_.itemData());
    case Opcode::SERVER_HPMP_UPDATE:
      return std::make_unique<ParsedServerHpMpUpdate>(packet);
    case Opcode::SERVER_ITEM_USE:
      return std::make_unique<ParsedServerUseItem>(packet);
    case Opcode::SERVER_STATS:
      return std::make_unique<ParsedServerAgentCharacterUpdateStats>(packet);
    case Opcode::CLIENT_AUTH:
    case Opcode::LOGIN_SERVER_CAPTCHA:
    // case static_cast<Opcode>(0x2005):
    // case static_cast<Opcode>(0x6005):
      return std::make_unique<ParsedUnknown>(packet);
  }
  std::cout << "Warning! No packet parser found for opcode " << std::hex << (int)packet.opcode << std::dec << '\n';
  return nullptr;
}

} // namespace packet::parsing