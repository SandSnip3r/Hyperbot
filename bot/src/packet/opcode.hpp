#ifndef PACKET_OPCODE_HPP_
#define PACKET_OPCODE_HPP_

#include <cstdint>
#include <string_view>

#define PACKET_OPCODE_LIST(F) \
  F(Hack, 0xA003) \
  F(FrameworkMessageIdentify, 0x2001) \
  F(FrameworkAliveNotify, 0x2002) \
  F(FrameworkStateNotify, 0x2005) \
  F(FrameworkStateRequest, 0x6005) \
  F(ClientGatewayPatchRequest, 0x6100) \
  F(ClientGatewayShardListRequest, 0x6101) \
  F(ClientGatewayShardListPingRequest, 0x6106) \
  F(ClientGatewayLoginRequest, 0x6102) \
  F(LOGIN_CLIENT_ACCEPT_HANDSHAKE, 0x9000) \
  F(LOGIN_CLIENT_LAUNCHER, 0x6104) \
  F(ClientGatewayLoginIbuvAnswer, 0x6323) \
  F(ServerGatewayLoginIbuvChallenge, 0x2322) \
  F(ServerGatewayLoginIbuvResult, 0xA323) \
  F(LOGIN_SERVER_INFO, 0x2001) \
  F(LOGIN_SERVER_HANDSHAKE, 0x5000) \
  F(LOGIN_SERVER_PATCH_INFO, 0x600D) \
  F(LOGIN_SERVER_LAUNCHER, 0x600D) \
  F(ServerGatewayPatchResponse, 0xA100) \
  F(ServerGatewayShardListResponse, 0xA101) \
  F(ServerGatewayLoginResponse, 0xA102) \
  F(CLIENT_INFO, 0x2001) \
  F(CLIENT_ACCEPT_HANDSHAKE, 0x9000) \
  F(CLIENT_PATCH_REQUEST, 0x6100) \
  F(ClientAgentAuthRequest, 0x6103) \
  F(ClientAgentInventoryOperationRequest, 0x7034) \
  F(ClientAgentGameResetComplete, 0x34B6) \
  F(ClientAgentGameReady, 0x3012) \
  F(CLIENT_CLOSE, 0x7005) \
  F(CLIENT_COUNTDOWN_INTERRUPT, 0x7006) \
  F(ClientAgentCharacterSelectionActionRequest, 0x7007) \
  F(ClientAgentChatRequest, 0x7025) \
  F(ClientAgentCharacterSelectionJoinRequest, 0x7001) \
  F(ClientAgentActionDeselectRequest, 0x704B) \
  F(ClientAgentActionSelectRequest, 0x7045) \
  F(ClientAgentActionTalkRequest, 0x7046) \
  F(ClientAgentCharacterMoveRequest, 0x7021) \
  F(ClientAgentCosCommandRequest, 0x70C5) \
  F(ClientAgentActionCommandRequest, 0x7074) \
  F(ClientAgentCharacterIncreaseStrRequest, 0x7050) \
  F(ClientAgentCharacterIncreaseIntRequest, 0x7051) \
  F(CLIENT_CHARACTER_STATE, 0x704F) \
  F(ClientAgentCharacterResurrect, 0x3053) \
  F(ClientAgentSkillMasteryLearnRequest, 0x70A2) \
  F(ClientAgentSkillLearnRequest, 0x70A1) \
  F(CLIENT_EMOTION, 0x3091) \
  F(ClientAgentInventoryItemUseRequest, 0x704C) \
  F(CLIENT_HOTKEY_CHANGE, 0x7158) \
  F(CLIENT_TELEPORT, 0x705A) \
  F(CLIENT_PARTY_FORM, 0x7069) \
  F(CLIENT_PARTY_EDIT, 0x706A) \
  F(CLIENT_PARTY_DELETE, 0x706B) \
  F(CLIENT_PARTY_MATCHING, 0x706C) \
  F(CLIENT_PARTY_REQUEST, 0x706D) \
  F(CLIENT_PARTY_ACCEPT, 0x306E) \
  F(CLIENT_PARTY_INVITE, 0x7060) \
  F(CLIENT_PARTY_DISMISS, 0x7061) \
  F(CLIENT_PARTY_KICK, 0x7063) \
  F(CLIENT_ANIMATION_INVITE, 0x3080) \
  F(ClientAgentAlchemyElixirRequest, 0x7150) \
  F(ClientAgentAlchemyStoneRequest, 0x7151) \
  F(CLIENT_TRANSPORT_HOME, 0x70CB) \
  F(CLIENT_TRANSPORT_DELETE, 0x70CB) \
  F(ClientAgentInventoryStorageOpenRequest, 0x703C) \
  F(ClientAgentInventoryRepairRequest, 0x703E) \
  F(ClientAgentCharacterUpdateBodyStateRequest, 0x70A7) \
  F(SERVER_INFO, 0x2001) \
  F(SERVER_HANDSHAKE, 0x5000) \
  F(SERVER_PATCH_INFO, 0x600D) \
  F(ServerAgentAuthResponse, 0xA103) \
  F(ServerAgentCharacterSelectionActionResponse, 0xB007) \
  F(ServerAgentCharacterSelectionJoinResponse, 0xB001) \
  F(ServerAgentCharacterDataBegin, 0x34A5) \
  F(ServerAgentCharacterData, 0x3013) \
  F(ServerAgentCharacterDataEnd, 0x34A6) \
  F(ServerAgentEnvironmentCelestialPosition, 0x3020) \
  F(ServerAgentEntitySpawn, 0x3015) \
  F(ServerAgentEntityDespawn, 0x3016) \
  F(ServerAgentEntityGroupspawnBegin, 0x3017) \
  F(ServerAgentEntityGroupspawnData, 0x3019) \
  F(ServerAgentEntityGroupspawnEnd, 0x3018) \
  F(SERVER_ITEM_EQUIP, 0x3038) \
  F(SERVER_ITEM_UNEQUIP, 0x3039) \
  F(ServerAgentInventoryOperationResponse, 0xB034) \
  F(SERVER_ANIMATION_ITEM_PICKUP, 0x3036) \
  F(ServerAgentInventoryItemUseResponse, 0xB04C) \
  F(SERVER_ANIMATION_ITEM_USE, 0x305C) \
  F(ServerAgentInventoryEquipCountdownStart, 0x3041) \
  F(ServerAgentInventoryUpdateItem, 0x3040) \
  F(SERVER_QUIT_GAME, 0x300A) \
  F(SERVER_COUNTDOWN, 0xB005) \
  F(SERVER_COUNTDOWN_INTERRUPT, 0xB006) \
  F(ServerAgentEntityUpdatePoints, 0x304E) \
  F(ServerAgentCharacterUpdateStats, 0x303D) \
  F(ServerAgentCharacterIncreaseStrResponse, 0xB050) \
  F(ServerAgentCharacterIncreaseIntResponse, 0xB051) \
  F(ServerAgentEntitySyncPosition, 0x3028) \
  F(ServerAgentEntityUpdateState, 0x30BF) \
  F(ServerAgentEntityUpdateHwanLevel, 0x30DF) \
  F(ServerAgentEntityUpdateMoveSpeed, 0x30D0) \
  F(ServerAgentEntityRemoveOwnership, 0x304D) \
  F(ServerAgentEntityUpdateStatus, 0x3057) \
  F(ServerAgentEntityDamageEffect, 0x3058) \
  F(SERVER_ANIMATION_LEVEL_UP, 0x3054) \
  F(ServerAgentEntityUpdateExperience, 0x3056) \
  F(ServerAgentSkillMasteryLearnResponse, 0xB0A2) \
  F(ServerAgentSkillLearnResponse, 0xB0A1) \
  F(ServerAgentChatUpdate, 0x3026) \
  F(SERVER_CHAT_ACCEPT, 0xB025) \
  F(ServerAgentActionDeselectResponse, 0xB04B) \
  F(ServerAgentActionSelectResponse, 0xB045) \
  F(ServerAgentActionTalkResponse, 0xB046) \
  F(ServerAgentEntityUpdateMovement, 0xB021) \
  F(SERVER_UNIQUE, 0x300C) \
  F(ServerAgentCosData, 0x30C8) \
  F(SERVER_COS_SIT_UP, 0xB0CB) \
  F(SERVER_ANIMATION_COS_REMOVE_MENU, 0x30C9) \
  F(SERVER_COS_DELETE, 0xB0C6) \
  F(ServerEnvironmentWeather, 0x3809) \
  F(ServerAgentSkillBegin, 0xB070) \
  F(ServerAgentSkillEnd, 0xB071) \
  F(ServerAgentActionCommandResponse, 0xB074) \
  F(ServerAgentBuffAdd, 0xB0BD) \
  F(ServerAgentBuffLink, 0xB0BE) \
  F(ServerAgentBuffRemove, 0xB072) \
  F(ServerAgentResurrectOption, 0x3011) \
  F(ServerAgentAbnormalInfo, 0x30D2) \
  F(SERVER_PARTY_FORM, 0xB069) \
  F(SERVER_PARTY_EDIT, 0xB06A) \
  F(SERVER_PARTY_DELETE, 0xB06B) \
  F(SERVER_PARTY_MATCHING, 0xB06C) \
  F(SERVER_PARTY_ACCEPT, 0xB06D) \
  F(SERVER_PARTY_REQUEST, 0x706D) \
  F(SERVER_PARTY_NEW_PARTY, 0x3065) \
  F(SERVER_PARTY_CHANGES, 0x3864) \
  F(SERVER_PARTY_INVITE, 0xB060) \
  F(SERVER_ANIMATION_INVITE, 0x3080) \
  F(SERVER_SILK_AMOUNT, 0x3153) \
  F(SERVER_TELEPORT, 0xB05A) \
  F(ServerAgentGameReset, 0x34B5) \
  F(ServerAgentInventoryStorageBegin, 0x3047) \
  F(ServerAgentInventoryStorageData, 0x3049) \
  F(ServerAgentInventoryStorageEnd, 0x3048) \
  F(ServerAgentGuildStorageBegin, 0x3253) \
  F(ServerAgentGuildStorageData, 0x3255) \
  F(ServerAgentGuildStorageEnd, 0x3254) \
  F(ServerAgentAlchemyElixirResponse, 0xB150) \
  F(ServerAgentAlchemyStoneResponse, 0xB151) \
  F(ServerAgentInventoryRepairResponse, 0xB03E) \
  F(ServerAgentInventoryUpdateDurability, 0x3052) \
  F(ServerAgentEntityUpdatePosition, 0xB023) \
  F(ServerAgentEntityUpdateAngle, 0xB024) \
  F(ClientAgentFreePvpUpdateRequest, 0x7516) \
  F(ServerAgentFreePvpUpdateResponse, 0xB516) \
  F(ClientAgentOperatorRequest, 0x7010) \
  F(ServerAgentOperatorResponse, 0xB010)

namespace packet {

enum class Opcode : uint16_t {
#define F(name, value) k##name = value,
  PACKET_OPCODE_LIST(F)
#undef F
};

std::string_view toString(Opcode opcode);

} // namespace packet

#endif // PACKET_OPCODE_HPP_