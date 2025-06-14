#include "packetParser.hpp"

#include "clientAgentActionCommandRequest.hpp"
#include "clientAgentActionDeselectRequest.hpp"
#include "clientAgentActionSelectRequest.hpp"
#include "clientAgentActionTalkRequest.hpp"
#include "clientAgentCharacterMoveRequest.hpp"
#include "clientAgentChatRequest.hpp"
#include "clientAgentInventoryItemUseRequest.hpp"
#include "frameworkMessageIdentify.hpp"
#include "serverAgentAbnormalInfo.hpp"
#include "serverAgentActionCommandResponse.hpp"
#include "serverAgentActionDeselectResponse.hpp"
#include "serverAgentActionSelectResponse.hpp"
#include "serverAgentActionTalkResponse.hpp"
#include "serverAgentAlchemyElixirResponse.hpp"
#include "serverAgentAlchemyStoneResponse.hpp"
#include "serverAgentAuthResponse.hpp"
#include "serverAgentBuffAdd.hpp"
#include "serverAgentBuffLink.hpp"
#include "serverAgentBuffRemove.hpp"
#include "serverAgentCharacterData.hpp"
#include "serverAgentCharacterIncreaseStrResponse.hpp"
#include "serverAgentCharacterIncreaseIntResponse.hpp"
#include "serverAgentCharacterSelectionActionResponse.hpp"
#include "serverAgentCharacterSelectionJoinResponse.hpp"
#include "serverAgentCharacterUpdateStats.hpp"
#include "serverAgentChatUpdate.hpp"
#include "serverAgentCosData.hpp"
#include "serverAgentEntityDamageEffect.hpp"
#include "serverAgentEntityDespawn.hpp"
#include "serverAgentEntityGroupSpawnData.hpp"
#include "serverAgentEntityRemoveOwnership.hpp"
#include "serverAgentEntitySpawn.hpp"
#include "serverAgentEntitySyncPosition.hpp"
#include "serverAgentEntityUpdateAngle.hpp"
#include "serverAgentEntityUpdateExperience.hpp"
#include "serverAgentEntityUpdateMovement.hpp"
#include "serverAgentEntityUpdateMoveSpeed.hpp"
#include "serverAgentEntityUpdateHwanLevel.hpp"
#include "serverAgentEntityUpdatePoints.hpp"
#include "serverAgentEntityUpdatePosition.hpp"
#include "serverAgentEntityUpdateState.hpp"
#include "serverAgentEntityUpdateStatus.hpp"
#include "serverAgentFreePvpUpdateResponse.hpp"
#include "serverAgentGameReset.hpp"
#include "serverAgentGuildStorageData.hpp"
#include "serverAgentInventoryEquipCountdownStart.hpp"
#include "serverAgentInventoryItemUseResponse.hpp"
#include "serverAgentInventoryOperationResponse.hpp"
#include "serverAgentInventoryRepairResponse.hpp"
#include "serverAgentInventoryStorageData.hpp"
#include "serverAgentInventoryUpdateDurability.hpp"
#include "serverAgentInventoryUpdateItem.hpp"
#include "serverAgentOperatorResponse.hpp"
#include "serverAgentResurrectOption.hpp"
#include "serverAgentSkillBegin.hpp"
#include "serverAgentSkillEnd.hpp"
#include "serverAgentSkillLearnResponse.hpp"
#include "serverAgentSkillMasteryLearnResponse.hpp"
#include "serverGatewayLoginIbuvChallenge.hpp"
#include "serverGatewayLoginResponse.hpp"
#include "serverGatewayPatchResponse.hpp"
#include "serverGatewayShardListResponse.hpp"

#include "packet/opcode.hpp"

#include <absl/log/log.h>

namespace packet::parsing {

PacketParser::PacketParser(const state::EntityTracker &entityTracker, const sro::pk2::GameData &gameData) :
      entityTracker_(entityTracker),
      gameData_(gameData) {
  //
}

std::unique_ptr<ParsedPacket> PacketParser::parsePacket(const PacketContainer &packet) const {
  // Given a packet's opcode, determine which parsed packet type is appropriate
  try {
    const auto opcode = static_cast<Opcode>(packet.opcode);
    switch (opcode) {
      case Opcode::kFrameworkMessageIdentify:
        return std::make_unique<FrameworkMessageIdentify>(packet);
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
      case Opcode::kServerGatewayPatchResponse:
        return std::make_unique<ServerGatewayPatchResponse>(packet);
      case Opcode::kServerGatewayShardListResponse:
        return std::make_unique<ServerGatewayShardListResponse>(packet);
      case Opcode::kServerGatewayLoginResponse:
        return std::make_unique<ServerGatewayLoginResponse>(packet);
      case Opcode::kServerAgentAuthResponse:
        return std::make_unique<ServerAgentAuthResponse>(packet);
      case Opcode::kServerAgentAlchemyElixirResponse:
        return std::make_unique<ServerAgentAlchemyElixirResponse>(packet, gameData_.itemData());
      case Opcode::kServerAgentAlchemyStoneResponse:
        return std::make_unique<ServerAgentAlchemyStoneResponse>(packet, gameData_.itemData());
      case Opcode::kServerAgentCharacterSelectionActionResponse:
        return std::make_unique<ServerAgentCharacterSelectionActionResponse>(packet);
      case Opcode::kServerAgentCharacterSelectionJoinResponse:
        return std::make_unique<ServerAgentCharacterSelectionJoinResponse>(packet);
      case Opcode::kServerAgentCharacterData:
        return std::make_unique<ServerAgentCharacterData>(packet, gameData_.itemData(), gameData_.skillData());
      case Opcode::kServerAgentEntityGroupspawnData:
        return std::make_unique<ServerAgentEntityGroupSpawnData>(packet, gameData_.characterData(), gameData_.itemData(), gameData_.skillData(), gameData_.teleportData());
      case Opcode::kServerAgentInventoryStorageData:
        return std::make_unique<ParsedServerAgentInventoryStorageData>(packet, gameData_.itemData());
      case Opcode::kServerAgentGuildStorageData:
        return std::make_unique<ServerAgentGuildStorageData>(packet, gameData_.itemData());
      case Opcode::kServerAgentEntitySpawn:
        return std::make_unique<ServerAgentEntitySpawn>(packet, gameData_.characterData(), gameData_.itemData(), gameData_.skillData(), gameData_.teleportData());
      case Opcode::kServerAgentEntityDespawn:
        return std::make_unique<ServerAgentEntityDespawn>(packet);
      case Opcode::kServerAgentEntityUpdateStatus:
        return std::make_unique<ServerAgentEntityUpdateStatus>(packet);
      case Opcode::kServerAgentEntityDamageEffect:
        return std::make_unique<ServerAgentEntityDamageEffect>(packet);
      case Opcode::kServerAgentEntityUpdateExperience:
        return std::make_unique<ServerAgentEntityUpdateExperience>(packet);
      case Opcode::kServerAgentAbnormalInfo:
        return std::make_unique<ServerAgentAbnormalInfo>(packet);
      case Opcode::kServerAgentInventoryUpdateItem:
        return std::make_unique<ServerAgentInventoryUpdateItem>(packet);
      case Opcode::kServerAgentEntityUpdatePoints:
        return std::make_unique<ServerAgentEntityUpdatePoints>(packet);
      case Opcode::kServerAgentCharacterUpdateStats:
        return std::make_unique<ServerAgentCharacterUpdateStats>(packet);
      case Opcode::kServerAgentCharacterIncreaseStrResponse:
        return std::make_unique<ServerAgentCharacterIncreaseStrResponse>(packet);
      case Opcode::kServerAgentCharacterIncreaseIntResponse:
        return std::make_unique<ServerAgentCharacterIncreaseIntResponse>(packet);
      case Opcode::kServerAgentInventoryItemUseResponse:
        return std::make_unique<ServerAgentInventoryItemUseResponse>(packet);
      case Opcode::kClientAgentInventoryItemUseRequest:
        return std::make_unique<ClientAgentInventoryItemUseRequest>(packet);
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
        return std::make_unique<ServerAgentActionSelectResponse>(packet, entityTracker_);
      case Opcode::kServerAgentActionTalkResponse:
        return std::make_unique<ServerAgentActionTalkResponse>(packet);
      case Opcode::kServerAgentEntityUpdateState:
        return std::make_unique<ServerAgentEntityUpdateState>(packet);
      case Opcode::kServerAgentEntityUpdateHwanLevel:
        return std::make_unique<ServerAgentEntityUpdateHwanLevel>(packet);
      case Opcode::kServerAgentBuffAdd:
        return std::make_unique<ServerAgentBuffAdd>(packet, gameData_.skillData());
      case Opcode::kServerAgentBuffLink:
        return std::make_unique<ServerAgentBuffLink>(packet);
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
      case Opcode::kServerAgentEntityRemoveOwnership:
        return std::make_unique<ServerAgentEntityRemoveOwnership>(packet);
      case Opcode::kServerAgentChatUpdate:
        return std::make_unique<ServerAgentChatUpdate>(packet);
      case Opcode::kServerAgentGameReset:
        return std::make_unique<ServerAgentGameReset>(packet);
      case Opcode::kServerAgentEntityUpdatePosition:
        return std::make_unique<ServerAgentEntityUpdatePosition>(packet);
      case Opcode::kServerAgentEntityUpdateAngle:
        return std::make_unique<ServerAgentEntityUpdateAngle>(packet);
      case Opcode::kServerAgentEntitySyncPosition:
        return std::make_unique<ServerAgentEntitySyncPosition>(packet);
      case Opcode::kServerAgentCosData:
        return std::make_unique<ServerAgentCosData>(packet, gameData_.characterData(), gameData_.itemData());
      case Opcode::kServerGatewayLoginIbuvChallenge:
        return std::make_unique<ServerGatewayLoginIbuvChallenge>(packet);
      case Opcode::kServerAgentSkillLearnResponse:
        return std::make_unique<ServerAgentSkillLearnResponse>(packet);
      case Opcode::kServerAgentSkillMasteryLearnResponse:
        return std::make_unique<ServerAgentSkillMasteryLearnResponse>(packet);
      case Opcode::kServerAgentResurrectOption:
        return std::make_unique<ServerAgentResurrectOption>(packet);
      case Opcode::kServerAgentOperatorResponse:
        return std::make_unique<ServerAgentOperatorResponse>(packet);
      case Opcode::kServerAgentFreePvpUpdateResponse:
        return std::make_unique<ServerAgentFreePvpUpdateResponse>(packet);
      case Opcode::kServerAgentInventoryEquipCountdownStart:
        return std::make_unique<ServerAgentInventoryEquipCountdownStart>(packet);
    }
    VLOG(2) << "No packet parser found for opcode " << std::hex << (int)packet.opcode << std::dec << '(' << toString(opcode) << ")";
  } catch (std::exception &ex) {
    LOG(ERROR) << "Exception while parsing packet! \"" << ex.what() << '"';
  }
  return nullptr;
}

} // namespace packet::parsing