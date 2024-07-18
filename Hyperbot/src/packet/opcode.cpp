#include "opcode.hpp"

namespace packet {

std::string toString(Opcode opcode) {
  if (opcode == Opcode::kServerEnvironmentWeather) {
    return "ServerEnvironmentWeather";
  }
  if (opcode == Opcode::HACK) {
    return "HACK";
  }
  if (opcode == Opcode::LOGIN_CLIENT_INFO) {
    return "LOGIN_CLIENT_INFO";
  }
  if (opcode == Opcode::LOGIN_CLIENT_KEEP_ALIVE) {
    return "LOGIN_CLIENT_KEEP_ALIVE";
  }
  if (opcode == Opcode::kClientGatewayPatchRequest) {
    return "ClientGatewayPatchRequest";
  }
  if (opcode == Opcode::LOGIN_CLIENT_SERVERLIST_REQUEST) {
    return "LOGIN_CLIENT_SERVERLIST_REQUEST";
  }
  if (opcode == Opcode::kClientGatewayLoginRequest) {
    return "ClientGatewayLoginRequest";
  }
  if (opcode == Opcode::LOGIN_CLIENT_ACCEPT_HANDSHAKE) {
    return "LOGIN_CLIENT_ACCEPT_HANDSHAKE";
  }
  if (opcode == Opcode::LOGIN_CLIENT_LAUNCHER) {
    return "LOGIN_CLIENT_LAUNCHER";
  }
  if (opcode == Opcode::LOGIN_SERVER_INFO) {
    return "LOGIN_SERVER_INFO";
  }
  if (opcode == Opcode::LOGIN_SERVER_HANDSHAKE) {
    return "LOGIN_SERVER_HANDSHAKE";
  }
  if (opcode == Opcode::LOGIN_SERVER_PATCH_INFO) {
    return "LOGIN_SERVER_PATCH_INFO";
  }
  if (opcode == Opcode::LOGIN_SERVER_LAUNCHER) {
    return "LOGIN_SERVER_LAUNCHER";
  }
  if (opcode == Opcode::kServerGatewayShardListResponse) {
    return "ServerGatewayShardListResponse";
  }
  if (opcode == Opcode::kServerGatewayLoginResponse) {
    return "ServerGatewayLoginResponse";
  }
  if (opcode == Opcode::CLIENT_INFO) {
    return "CLIENT_INFO";
  }
  if (opcode == Opcode::CLIENT_ACCEPT_HANDSHAKE) {
    return "CLIENT_ACCEPT_HANDSHAKE";
  }
  if (opcode == Opcode::CLIENT_KEEP_ALIVE) {
    return "CLIENT_KEEP_ALIVE";
  }
  if (opcode == Opcode::CLIENT_PATCH_REQUEST) {
    return "CLIENT_PATCH_REQUEST";
  }
  if (opcode == Opcode::kClientAgentAuthRequest) {
    return "ClientAgentAuthRequest";
  }
  if (opcode == Opcode::kClientAgentInventoryOperationRequest) {
    return "ClientAgentInventoryOperationRequest";
  }
  if (opcode == Opcode::CLIENT_INGAME_NOTIFY) {
    return "CLIENT_INGAME_NOTIFY";
  }
  if (opcode == Opcode::CLIENT_CLOSE) {
    return "CLIENT_CLOSE";
  }
  if (opcode == Opcode::CLIENT_COUNTDOWN_INTERRUPT) {
    return "CLIENT_COUNTDOWN_INTERRUPT";
  }
  if (opcode == Opcode::kClientAgentCharacterSelectionActionRequest) {
    return "ClientAgentCharacterSelectionActionRequest";
  }
  if (opcode == Opcode::kClientAgentChatRequest) {
    return "ClientAgentChatRequest";
  }
  if (opcode == Opcode::kClientAgentCharacterSelectionJoinRequest) {
    return "ClientAgentCharacterSelectionJoinRequest";
  }
  if (opcode == Opcode::kClientAgentActionSelectRequest) {
    return "ClientAgentActionSelectRequest";
  }
  if (opcode == Opcode::kClientAgentOperatorRequest) {
    return "ClientAgentOperatorRequest";
  }
  if (opcode == Opcode::kServerAgentOperatorResponse) {
    return "ServerAgentOperatorResponse";
  }
  if (opcode == Opcode::kClientAgentCharacterMoveRequest) {
    return "ClientAgentCharacterMoveRequest";
  }
  if (opcode == Opcode::kClientAgentCosCommandRequest) {
    return "ClientAgentCosCommandRequest";
  }
  if (opcode == Opcode::kClientAgentActionCommandRequest) {
    return "ClientAgentActionCommandRequest";
  }
  if (opcode == Opcode::kClientAgentCharacterIncreaseStrRequest) {
    return "ClientAgentCharacterIncreaseStrRequest";
  }
  if (opcode == Opcode::kClientAgentCharacterIncreaseIntRequest) {
    return "ClientAgentCharacterIncreaseIntRequest";
  }
  if (opcode == Opcode::CLIENT_CHARACTER_STATE) {
    return "CLIENT_CHARACTER_STATE";
  }
  if (opcode == Opcode::kClientAgentCharacterResurrect) {
    return "ClientAgentCharacterResurrect";
  }
  if (opcode == Opcode::CLIENT_MASTERYUPDATE) {
    return "CLIENT_MASTERYUPDATE";
  }
  if (opcode == Opcode::CLIENT_SKILLUPDATE) {
    return "CLIENT_SKILLUPDATE";
  }
  if (opcode == Opcode::CLIENT_EMOTION) {
    return "CLIENT_EMOTION";
  }
  if (opcode == Opcode::kClientAgentInventoryItemUseRequest) {
    return "ClientAgentInventoryItemUseRequest";
  }
  if (opcode == Opcode::CLIENT_HOTKEY_CHANGE) {
    return "CLIENT_HOTKEY_CHANGE";
  }
  if (opcode == Opcode::kClientAgentActionTalkRequest) {
    return "ClientAgentActionTalkRequest";
  }
  if (opcode == Opcode::kClientAgentActionDeselectRequest) {
    return "ClientAgentActionDeselectRequest";
  }
  if (opcode == Opcode::CLIENT_TELEPORT) {
    return "CLIENT_TELEPORT";
  }
  if (opcode == Opcode::CLIENT_PARTY_FORM) {
    return "CLIENT_PARTY_FORM";
  }
  if (opcode == Opcode::CLIENT_PARTY_EDIT) {
    return "CLIENT_PARTY_EDIT";
  }
  if (opcode == Opcode::CLIENT_PARTY_DELETE) {
    return "CLIENT_PARTY_DELETE";
  }
  if (opcode == Opcode::CLIENT_PARTY_MATCHING) {
    return "CLIENT_PARTY_MATCHING";
  }
  if (opcode == Opcode::CLIENT_PARTY_REQUEST) {
    return "CLIENT_PARTY_REQUEST";
  }
  if (opcode == Opcode::CLIENT_PARTY_ACCEPT) {
    return "CLIENT_PARTY_ACCEPT";
  }
  if (opcode == Opcode::CLIENT_PARTY_INVITE) {
    return "CLIENT_PARTY_INVITE";
  }
  if (opcode == Opcode::CLIENT_PARTY_DISMISS) {
    return "CLIENT_PARTY_DISMISS";
  }
  if (opcode == Opcode::CLIENT_PARTY_KICK) {
    return "CLIENT_PARTY_KICK";
  }
  if (opcode == Opcode::CLIENT_ANIMATION_INVITE) {
    return "CLIENT_ANIMATION_INVITE";
  }
  if (opcode == Opcode::kClientAgentAlchemyElixirRequest) {
    return "ClientAgentAlchemyElixirRequest";
  }
  if (opcode == Opcode::kClientAgentAlchemyStoneRequest) {
    return "ClientAgentAlchemyStoneRequest";
  }
  if (opcode == Opcode::CLIENT_TRANSPORT_HOME) {
    return "CLIENT_TRANSPORT_HOME";
  }
  if (opcode == Opcode::CLIENT_TRANSPORT_DELETE) {
    return "CLIENT_TRANSPORT_DELETE";
  }
  if (opcode == Opcode::kClientAgentInventoryStorageOpenRequest) {
    return "ClientAgentInventoryStorageOpenRequest";
  }
  if (opcode == Opcode::kClientAgentInventoryRepairRequest) {
    return "ClientAgentInventoryRepairRequest";
  }
  if (opcode == Opcode::kClientAgentCharacterUpdateBodyStateRequest) {
    return "ClientAgentCharacterUpdateBodyStateRequest";
  }
  if (opcode == Opcode::SERVER_INFO) {
    return "SERVER_INFO";
  }
  if (opcode == Opcode::SERVER_HANDSHAKE) {
    return "SERVER_HANDSHAKE";
  }
  if (opcode == Opcode::SERVER_PATCH_INFO) {
    return "SERVER_PATCH_INFO";
  }
  if (opcode == Opcode::kServerAgentAuthResponse) {
    return "ServerAgentAuthResponse";
  }
  if (opcode == Opcode::kServerAgentCharacterSelectionActionResponse) {
    return "ServerAgentCharacterSelectionActionResponse";
  }
  if (opcode == Opcode::kServerAgentCharacterData) {
    return "ServerAgentCharacterData";
  }
  if (opcode == Opcode::kServerAgentCharacterSelectionJoinResponse) {
    return "ServerAgentCharacterSelectionJoinResponse";
  }
  if (opcode == Opcode::SERVER_AGENT_CHARACTER_INFO_BEGIN) {
    return "SERVER_AGENT_CHARACTER_INFO_BEGIN";
  }
  if (opcode == Opcode::SERVER_AGENT_CHARACTER_INFO_END) {
    return "SERVER_AGENT_CHARACTER_INFO_END";
  }
  if (opcode == Opcode::kServerAgentEnvironmentCelestialPosition) {
    return "ServerAgentEnvironmentCelestialPosition";
  }
  if (opcode == Opcode::kServerAgentEntitySpawn) {
    return "ServerAgentEntitySpawn";
  }
  if (opcode == Opcode::kServerAgentEntityDespawn) {
    return "ServerAgentEntityDespawn";
  }
  if (opcode == Opcode::kServerAgentEntityGroupspawnBegin) {
    return "ServerAgentEntityGroupspawnBegin";
  }
  if (opcode == Opcode::kServerAgentEntityGroupspawnData) {
    return "ServerAgentEntityGroupspawnData";
  }
  if (opcode == Opcode::kServerAgentEntityGroupspawnEnd) {
    return "ServerAgentEntityGroupspawnEnd";
  }
  if (opcode == Opcode::SERVER_ITEM_EQUIP) {
    return "SERVER_ITEM_EQUIP";
  }
  if (opcode == Opcode::SERVER_ITEM_UNEQUIP) {
    return "SERVER_ITEM_UNEQUIP";
  }
  if (opcode == Opcode::kServerAgentInventoryOperationResponse) {
    return "ServerAgentInventoryOperationResponse";
  }
  if (opcode == Opcode::SERVER_ANIMATION_ITEM_PICKUP) {
    return "SERVER_ANIMATION_ITEM_PICKUP";
  }
  if (opcode == Opcode::kServerAgentInventoryItemUseResponse) {
    return "ServerAgentInventoryItemUseResponse";
  }
  if (opcode == Opcode::SERVER_ANIMATION_ITEM_USE) {
    return "SERVER_ANIMATION_ITEM_USE";
  }
  if (opcode == Opcode::SERVER_ANIMATION_CAPE) {
    return "SERVER_ANIMATION_CAPE";
  }
  if (opcode == Opcode::kServerAgentInventoryUpdateItem) {
    return "ServerAgentInventoryUpdateItem";
  }
  if (opcode == Opcode::SERVER_QUIT_GAME) {
    return "SERVER_QUIT_GAME";
  }
  if (opcode == Opcode::SERVER_COUNTDOWN) {
    return "SERVER_COUNTDOWN";
  }
  if (opcode == Opcode::SERVER_COUNTDOWN_INTERRUPT) {
    return "SERVER_COUNTDOWN_INTERRUPT";
  }
  if (opcode == Opcode::kServerAgentEntityUpdatePoints) {
    return "ServerAgentEntityUpdatePoints";
  }
  if (opcode == Opcode::kServerAgentCharacterUpdateStats) {
    return "ServerAgentCharacterUpdateStats";
  }
  if (opcode == Opcode::kServerAgentCharacterIncreaseStrResponse) {
    return "ServerAgentCharacterIncreaseStrResponse";
  }
  if (opcode == Opcode::kServerAgentCharacterIncreaseIntResponse) {
    return "ServerAgentCharacterIncreaseIntResponse";
  }
  if (opcode == Opcode::kServerAgentEntityUpdateState) {
    return "ServerAgentEntityUpdateState";
  }
  if (opcode == Opcode::kServerAgentEntityUpdateHwanLevel) {
    return "ServerAgentEntityUpdateHwanLevel";
  }
  if (opcode == Opcode::kServerAgentEntityUpdateMoveSpeed) {
    return "ServerAgentEntityUpdateMoveSpeed";
  }
  if (opcode == Opcode::kServerAgentEntityRemoveOwnership) {
    return "ServerAgentEntityRemoveOwnership";
  }
  if (opcode == Opcode::kServerAgentEntityUpdateStatus) {
    return "ServerAgentEntityUpdateStatus";
  }
  if (opcode == Opcode::kServerAgentEntityDamageEffect) {
    return "ServerAgentEntityDamageEffect";
  }
  if (opcode == Opcode::SERVER_ANIMATION_LEVEL_UP) {
    return "SERVER_ANIMATION_LEVEL_UP";
  }
  if (opcode == Opcode::kServerAgentEntityUpdateExperience) {
    return "ServerAgentEntityUpdateExperience";
  }
  if (opcode == Opcode::kServerAgentSkillMasteryLearnResponse) {
    return "ServerAgentSkillMasteryLearnResponse";
  }
  if (opcode == Opcode::kServerAgentSkillLearnResponse) {
    return "ServerAgentSkillLearnResponse";
  }
  if (opcode == Opcode::kServerAgentChatUpdate) {
    return "ServerAgentChatUpdate";
  }
  if (opcode == Opcode::SERVER_CHAT_ACCEPT) {
    return "SERVER_CHAT_ACCEPT";
  }
  if (opcode == Opcode::kServerAgentActionDeselectResponse) {
    return "ServerAgentActionDeselectResponse";
  }
  if (opcode == Opcode::kServerAgentActionSelectResponse) {
    return "ServerAgentActionSelectResponse";
  }
  if (opcode == Opcode::kServerAgentActionTalkResponse) {
    return "ServerAgentActionTalkResponse";
  }
  if (opcode == Opcode::kServerAgentEntityUpdateMovement) {
    return "ServerAgentEntityUpdateMovement";
  }
  if (opcode == Opcode::SERVER_UNIQUE) {
    return "SERVER_UNIQUE";
  }
  if (opcode == Opcode::kServerAgentCosData) {
    return "ServerAgentCosData";
  }
  if (opcode == Opcode::SERVER_COS_SIT_UP) {
    return "SERVER_COS_SIT_UP";
  }
  if (opcode == Opcode::SERVER_ANIMATION_COS_REMOVE_MENU) {
    return "SERVER_ANIMATION_COS_REMOVE_MENU";
  }
  if (opcode == Opcode::SERVER_COS_DELETE) {
    return "SERVER_COS_DELETE";
  }
  if (opcode == Opcode::kServerAgentSkillBegin) {
    return "ServerAgentSkillBegin";
  }
  if (opcode == Opcode::kServerAgentActionCommandResponse) {
    return "ServerAgentActionCommandResponse";
  }
  if (opcode == Opcode::kServerAgentSkillEnd) {
    return "ServerAgentSkillEnd";
  }
  if (opcode == Opcode::kServerAgentBuffAdd) {
    return "ServerAgentBuffAdd";
  }
  if (opcode == Opcode::kServerAgentBuffLink) {
    return "ServerAgentBuffLink";
  }
  if (opcode == Opcode::kServerAgentBuffRemove) {
    return "ServerAgentBuffRemove";
  }
  if (opcode == Opcode::kServerAgentResurrectOption) {
    return "ServerAgentResurrectOption";
  }
  if (opcode == Opcode::kServerAgentAbnormalInfo) {
    return "ServerAgentAbnormalInfo";
  }
  if (opcode == Opcode::SERVER_PARTY_FORM) {
    return "SERVER_PARTY_FORM";
  }
  if (opcode == Opcode::SERVER_PARTY_EDIT) {
    return "SERVER_PARTY_EDIT";
  }
  if (opcode == Opcode::SERVER_PARTY_DELETE) {
    return "SERVER_PARTY_DELETE";
  }
  if (opcode == Opcode::SERVER_PARTY_MATCHING) {
    return "SERVER_PARTY_MATCHING";
  }
  if (opcode == Opcode::SERVER_PARTY_ACCEPT) {
    return "SERVER_PARTY_ACCEPT";
  }
  if (opcode == Opcode::SERVER_PARTY_REQUEST) {
    return "SERVER_PARTY_REQUEST";
  }
  if (opcode == Opcode::SERVER_PARTY_NEW_PARTY) {
    return "SERVER_PARTY_NEW_PARTY";
  }
  if (opcode == Opcode::SERVER_PARTY_CHANGES) {
    return "SERVER_PARTY_CHANGES";
  }
  if (opcode == Opcode::SERVER_PARTY_INVITE) {
    return "SERVER_PARTY_INVITE";
  }
  if (opcode == Opcode::SERVER_ANIMATION_INVITE) {
    return "SERVER_ANIMATION_INVITE";
  }
  if (opcode == Opcode::SERVER_SILK_AMOUNT) {
    return "SERVER_SILK_AMOUNT";
  }
  if (opcode == Opcode::SERVER_TELEPORT) {
    return "SERVER_TELEPORT";
  }
  if (opcode == Opcode::kServerAgentGameReset) {
    return "ServerAgentGameReset";
  }
  if (opcode == Opcode::kServerAgentInventoryStorageBegin) {
    return "ServerAgentInventoryStorageBegin";
  }
  if (opcode == Opcode::kServerAgentInventoryStorageData) {
    return "ServerAgentInventoryStorageData";
  }
  if (opcode == Opcode::kServerAgentInventoryStorageEnd) {
    return "ServerAgentInventoryStorageEnd";
  }
  if (opcode == Opcode::kServerAgentAlchemyElixirResponse) {
    return "ServerAgentAlchemyElixirResponse";
  }
  if (opcode == Opcode::kServerAgentAlchemyStoneResponse) {
    return "ServerAgentAlchemyStoneResponse";
  }
  if (opcode == Opcode::kServerAgentInventoryRepairResponse) {
    return "ServerAgentInventoryRepairResponse";
  }
  if (opcode == Opcode::kServerAgentInventoryUpdateDurability) {
    return "ServerAgentInventoryUpdateDurability";
  }
  if (opcode == Opcode::kServerAgentEntityUpdatePosition) {
    return "ServerAgentEntityUpdatePosition";
  }
  if (opcode == Opcode::kClientAgentFreePvpUpdateRequest) {
    return "ClientAgentFreePvpUpdateRequest";
  }
  if (opcode == Opcode::kServerAgentFreePvpUpdateResponse) {
    return "ServerAgentFreePvpUpdateResponse";
  }
  if (opcode == Opcode::kClientGatewayLoginIbuvAnswer) {
    return "ClientGatewayLoginIbuvAnswer";
  }
  if (opcode == Opcode::kServerGatewayLoginIbuvChallenge) {
    return "ServerGatewayLoginIbuvChallenge";
  }
  if (opcode == Opcode::kServerGatewayLoginIbuvResult) {
    return "ServerGatewayLoginIbuvResult";
  }
  return "UNKNOWN-"+std::to_string(static_cast<int>(opcode));
}

} // namespace packet