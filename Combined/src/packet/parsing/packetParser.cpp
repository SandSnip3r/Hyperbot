#include "clientAgentActionCommandRequest.hpp"
#include "clientAgentChatRequest.hpp"
#include "packetParser.hpp"
#include "serverAgentActionCommandResponse.hpp"
#include "serverAgentActionSelectResponse.hpp"
#include "serverAgentBuffAdd.hpp"
#include "serverAgentBuffRemove.hpp"
#include "serverAgentCharacterData.hpp"
#include "serverAgentChatUpdate.hpp"
#include "serverAgentEntityUpdateState.hpp"
#include "serverAgentSkillBegin.hpp"
#include "serverAgentSkillEnd.hpp"
#include "../opcode.hpp"

#include <iostream>

namespace packet::parsing {

PacketParser::PacketParser(const pk2::GameData &gameData) :
      gameData_(gameData) {
  //
}

std::unique_ptr<ParsedPacket> PacketParser::parsePacket(const PacketContainer &packet) const {
  // Given a packet's opcode, determine which parsed packet type is appropriate
  switch (static_cast<Opcode>(packet.opcode)) {
    case Opcode::kClientAgentChatRequest:
      return std::make_unique<ClientAgentChatRequest>(packet);
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
    case Opcode::kServerAgentCharacterData:
      return std::make_unique<ParsedServerAgentCharacterData>(packet, gameData_.itemData(), gameData_.skillData());
    case Opcode::SERVER_AGENT_ENTITY_GROUPSPAWN_DATA:
      return std::make_unique<ParsedServerAgentGroupSpawn>(packet, gameData_.characterData(), gameData_.itemData(), gameData_.skillData(), gameData_.teleportData());
    case Opcode::SERVER_SPAWN:
      return std::make_unique<ParsedServerAgentSpawn>(packet, gameData_.characterData(), gameData_.itemData(), gameData_.skillData(), gameData_.teleportData());
    case Opcode::SERVER_DESPAWN:
      return std::make_unique<ParsedServerAgentDespawn>(packet);
    case Opcode::SERVER_HPMP_UPDATE:
      return std::make_unique<ParsedServerHpMpUpdate>(packet);
    case Opcode::SERVER_AGENT_ABNORMAL_INFO:
      return std::make_unique<ParsedServerAbnormalInfo>(packet);
    case Opcode::SERVER_ITEM_USE:
      return std::make_unique<ParsedServerUseItem>(packet);
    case Opcode::SERVER_STATS:
      return std::make_unique<ParsedServerAgentCharacterUpdateStats>(packet);
    case Opcode::SERVER_ITEM_MOVEMENT:
      return std::make_unique<ParsedServerItemMove>(packet, gameData_.itemData());
    case Opcode::kClientAgentInventoryOperationRequest:
      return std::make_unique<ParsedClientItemMove>(packet);
    case Opcode::kServerAgentActionCommandResponse:
      return std::make_unique<ServerAgentActionCommandResponse>(packet);
    case Opcode::kServerAgentActionSelectResponse:
      return std::make_unique<ServerAgentActionSelectResponse>(packet);
    case Opcode::kServerAgentEntityUpdateState:
      return std::make_unique<ServerAgentEntityUpdateState>(packet);
    case Opcode::kServerAgentBuffAdd:
      return std::make_unique<ServerAgentBuffAdd>(packet, gameData_.skillData());
    case Opcode::kServerAgentBuffRemove:
      return std::make_unique<ServerAgentBuffRemove>(packet);
    case Opcode::kClientAgentActionCommandRequest:
      return std::make_unique<ClientAgentActionCommandRequest>(packet);
    case Opcode::kServerAgentSkillBegin:
      return std::make_unique<ServerAgentSkillBegin>(packet);
    case Opcode::kServerAgentSkillEnd:
      return std::make_unique<ServerAgentSkillEnd>(packet);
    case Opcode::kServerAgentChatUpdate:
      return std::make_unique<ServerAgentChatUpdate>(packet);
    case Opcode::kClientAgentAuthRequest:
    case Opcode::kServerGatewayLoginIbuvChallenge:
    // case static_cast<Opcode>(0x2005):
    // case static_cast<Opcode>(0x6005):
      return std::make_unique<ParsedUnknown>(packet);
  }
  std::cout << "Warning! No packet parser found for opcode " << std::hex << (int)packet.opcode << std::dec << '\n';
  return nullptr;
}

} // namespace packet::parsing