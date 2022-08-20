#include "packetParser.hpp"

#include "clientAgentActionCommandRequest.hpp"
#include "clientAgentActionDeselectRequest.hpp"
#include "clientAgentActionSelectRequest.hpp"
#include "clientAgentActionTalkRequest.hpp"
#include "clientAgentCharacterMoveRequest.hpp"
#include "clientAgentChatRequest.hpp"
#include "serverAgentActionCommandResponse.hpp"
#include "serverAgentActionDeselectResponse.hpp"
#include "serverAgentActionSelectResponse.hpp"
#include "serverAgentActionTalkResponse.hpp"
#include "serverAgentBuffAdd.hpp"
#include "serverAgentBuffRemove.hpp"
#include "serverAgentCharacterData.hpp"
#include "serverAgentChatUpdate.hpp"
#include "serverAgentEntitySyncPosition.hpp"
#include "serverAgentEntityUpdateExperience.hpp"
#include "serverAgentEntityUpdateMovement.hpp"
#include "serverAgentEntityUpdateMoveSpeed.hpp"
#include "serverAgentEntityUpdatePoints.hpp"
#include "serverAgentEntityUpdatePosition.hpp"
#include "serverAgentEntityUpdateState.hpp"
#include "serverAgentInventoryOperationResponse.hpp"
#include "serverAgentInventoryRepairResponse.hpp"
#include "serverAgentInventoryStorageData.hpp"
#include "serverAgentInventoryUpdateDurability.hpp"
#include "serverAgentSkillBegin.hpp"
#include "serverAgentSkillEnd.hpp"

#include "packet/opcode.hpp"

#include <iostream>

namespace packet::parsing {

PacketParser::PacketParser(const pk2::GameData &gameData) :
      gameData_(gameData) {
  //
}

std::unique_ptr<ParsedPacket> PacketParser::parsePacket(const PacketContainer &packet) const {
  // Given a packet's opcode, determine which parsed packet type is appropriate
  try {
    switch (static_cast<Opcode>(packet.opcode)) {
      case Opcode::kClientAgentChatRequest:
        return std::make_unique<ClientAgentChatRequest>(packet);
      case Opcode::kClientAgentCharacterMoveRequest:
        return std::make_unique<ClientAgentCharacterMoveRequest>(packet);
      case Opcode::kServerAgentEntityUpdateMovement:
        return std::make_unique<ServerAgentEntityUpdateMovement>(packet);
      case Opcode::kClientAgentActionDeselectRequest:
        return std::make_unique<ClientAgentActionDeselectRequest>(packet);
      case Opcode::kClientAgentActionSelectRequest:
        return std::make_unique<ClientAgentActionSelectRequest>(packet);
      case Opcode::kClientAgentActionTalkRequest:
        return std::make_unique<ClientAgentActionTalkRequest>(packet);
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
      case Opcode::kServerAgentEntityGroupspawnData:
        return std::make_unique<ParsedServerAgentEntityGroupSpawnData>(packet, gameData_.characterData(), gameData_.itemData(), gameData_.skillData(), gameData_.teleportData());
      case Opcode::kServerAgentInventoryStorageData:
        return std::make_unique<ParsedServerAgentInvetoryStorageData>(packet, gameData_.itemData());
      case Opcode::kServerAgentEntitySpawn:
        return std::make_unique<ParsedServerAgentSpawn>(packet, gameData_.characterData(), gameData_.itemData(), gameData_.skillData(), gameData_.teleportData());
      case Opcode::kServerAgentEntityDespawn:
        return std::make_unique<ParsedServerAgentDespawn>(packet);
      case Opcode::kServerAgentEntityUpdateStatus:
        return std::make_unique<ParsedServerAgentEntityUpdateStatus>(packet);
      case Opcode::kServerAgentEntityUpdateExperience:
        return std::make_unique<ServerAgentEntityUpdateExperience>(packet);
      case Opcode::kServerAgentAbnormalInfo:
        return std::make_unique<ParsedServerAgentAbnormalInfo>(packet);
      case Opcode::kServerAgentInventoryItemUseResponse:
        return std::make_unique<ParsedServerAgentInventoryItemUseResponse>(packet);
      case Opcode::kServerAgentEntityUpdatePoints:
        return std::make_unique<ServerAgentEntityUpdatePoints>(packet);
      case Opcode::kServerAgentCharacterUpdateStats:
        return std::make_unique<ParsedServerAgentCharacterUpdateStats>(packet);
      case Opcode::kServerAgentInventoryOperationResponse:
        return std::make_unique<ServerAgentInventoryOperationResponse>(packet, gameData_.itemData());
      case Opcode::kServerAgentInventoryRepairResponse:
        return std::make_unique<ServerAgentInventoryRepairResponse>(packet);
      case Opcode::kServerAgentInventoryUpdateDurability:
        return std::make_unique<ServerAgentInventoryUpdateDurability>(packet);
      case Opcode::kClientAgentInventoryOperationRequest:
        return std::make_unique<ParsedClientItemMove>(packet);
      case Opcode::kServerAgentActionCommandResponse:
        return std::make_unique<ServerAgentActionCommandResponse>(packet);
      case Opcode::kServerAgentActionDeselectResponse:
        return std::make_unique<ServerAgentActionDeselectResponse>(packet);
      case Opcode::kServerAgentActionSelectResponse:
        return std::make_unique<ServerAgentActionSelectResponse>(packet);
      case Opcode::kServerAgentActionTalkResponse:
        return std::make_unique<ServerAgentActionTalkResponse>(packet);
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
      case Opcode::kServerAgentEntityUpdateMoveSpeed:
        return std::make_unique<ServerAgentEntityUpdateMoveSpeed>(packet);
      case Opcode::kServerAgentChatUpdate:
        return std::make_unique<ServerAgentChatUpdate>(packet);
      case Opcode::kServerAgentEntityUpdatePosition:
        return std::make_unique<ServerAgentEntityUpdatePosition>(packet);
      case Opcode::kServerAgentEntitySyncPosition:
        return std::make_unique<ServerAgentEntitySyncPosition>(packet);
      case Opcode::kClientAgentAuthRequest:
      case Opcode::kServerGatewayLoginIbuvChallenge:
      // case static_cast<Opcode>(0x2005):
      // case static_cast<Opcode>(0x6005):
        return std::make_unique<ParsedUnknown>(packet);
    }
    std::cout << "Warning! No packet parser found for opcode " << std::hex << (int)packet.opcode << std::dec << '\n';
  } catch (std::exception &ex) {
    std::cout << "Exception while parsing packet!\n";
    std::cout << "  " << ex.what() << '\n';
  }
  return nullptr;
}

} // namespace packet::parsing