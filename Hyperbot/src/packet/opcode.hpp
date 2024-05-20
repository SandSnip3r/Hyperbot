#include <ostream>
#include <string>

#ifndef PACKET_OPCODE_HPP_
#define PACKET_OPCODE_HPP_

namespace packet {

// Game packets
//0x7___ Request
//0xB___ Ack
// Framework packets
//0x6___ Request
//0xA___ Ack
// Net Engine packets
//0x5___ Request
//0x9___ Ack
/*
0000 = App_NoDir
1000 = NetEngine_NoDir
2000 = Framework_NoDir
3000 = Game_NoDir
4000 = App_Req
5000 = NetEngine_Req
6000 = Framework_Req
7000 = Game_Req
8000 = App_Ack
9000 = NetEngine_Ack
A000 = Framework_Ack
B000 = Game_Ack
*/

enum class Opcode : uint16_t {
  HACK = 0xA003,

  LOGIN_CLIENT_INFO = 0x2001, // LOGIN_CLIENT_INFO, LOGIN_SERVER_INFO, CLIENT_INFO, SERVER_INFO
  LOGIN_CLIENT_KEEP_ALIVE = 0x2002, // LOGIN_CLIENT_KEEP_ALIVE, CLIENT_KEEP_ALIVE
  kClientGatewayPatchRequest = 0x6100, // kClientGatewayPatchRequest, CLIENT_PATCH_REQUEST
  LOGIN_CLIENT_SERVERLIST_REQUEST = 0x6101,
  kClientGatewayLoginRequest = 0x6102,
  LOGIN_CLIENT_ACCEPT_HANDSHAKE = 0x9000, // LOGIN_CLIENT_ACCEPT_HANDSHAKE, CLIENT_ACCEPT_HANDSHAKE
  LOGIN_CLIENT_LAUNCHER = 0x6104,

  kClientGatewayLoginIbuvAnswer = 0x6323,
  kServerGatewayLoginIbuvChallenge = 0x2322,
  kServerGatewayLoginIbuvResult = 0xA323,

  LOGIN_SERVER_INFO = 0x2001, // LOGIN_CLIENT_INFO, LOGIN_SERVER_INFO, CLIENT_INFO, SERVER_INFO
  LOGIN_SERVER_HANDSHAKE = 0x5000, // LOGIN_SERVER_HANDSHAKE, SERVER_HANDSHAKE
  LOGIN_SERVER_PATCH_INFO = 0x600D, // LOGIN_SERVER_PATCH_INFO, LOGIN_SERVER_LAUNCHER, SERVER_PATCH_INFO
  LOGIN_SERVER_LAUNCHER = 0x600D, // LOGIN_SERVER_PATCH_INFO, LOGIN_SERVER_LAUNCHER, SERVER_PATCH_INFO
  kServerGatewayShardListResponse = 0xA101,
  kServerGatewayLoginResponse = 0xA102,

  CLIENT_INFO = 0x2001, // LOGIN_CLIENT_INFO, LOGIN_SERVER_INFO, CLIENT_INFO, SERVER_INFO
  CLIENT_ACCEPT_HANDSHAKE = 0x9000, // LOGIN_CLIENT_ACCEPT_HANDSHAKE, CLIENT_ACCEPT_HANDSHAKE
  CLIENT_KEEP_ALIVE = 0x2002, // LOGIN_CLIENT_KEEP_ALIVE, CLIENT_KEEP_ALIVE
  CLIENT_PATCH_REQUEST = 0x6100, // kClientGatewayPatchRequest, CLIENT_PATCH_REQUEST
  kClientAgentAuthRequest = 0x6103,
  kClientAgentInventoryOperationRequest = 0x7034,
  CLIENT_INGAME_NOTIFY = 0x3012,//0x70EA,
  CLIENT_CLOSE = 0x7005,
  CLIENT_COUNTDOWN_INTERRUPT = 0x7006,
  kClientAgentCharacterSelectionActionRequest = 0x7007,
  kClientAgentChatRequest = 0x7025,
  kClientAgentCharacterSelectionJoinRequest = 0x7001,
  kClientAgentActionDeselectRequest = 0x704B,
  kClientAgentActionSelectRequest = 0x7045,
  kClientAgentActionTalkRequest = 0x7046,
  kClientAgentCharacterMoveRequest = 0x7021,
  kClientAgentCosCommandRequest = 0x70C5,
  kClientAgentActionCommandRequest = 0x7074,
  kClientAgentCharacterIncreaseStrRequest = 0x7050,
  kClientAgentCharacterIncreaseIntRequest = 0x7051,
  CLIENT_CHARACTER_STATE = 0x704F,
  kClientAgentCharacterResurrect = 0x3053,
  CLIENT_MASTERYUPDATE = 0x70A2,
  CLIENT_SKILLUPDATE = 0x70A1,
  CLIENT_EMOTION = 0x3091,
  kClientAgentInventoryItemUseRequest = 0x704C,
  CLIENT_HOTKEY_CHANGE = 0x7158,
  CLIENT_TELEPORT = 0x705A,
  CLIENT_PARTY_FORM = 0x7069,
  CLIENT_PARTY_EDIT = 0x706A,
  CLIENT_PARTY_DELETE = 0x706B,
  CLIENT_PARTY_MATCHING = 0x706C,
  CLIENT_PARTY_REQUEST = 0x706D,
  CLIENT_PARTY_ACCEPT = 0x306E,
  CLIENT_PARTY_INVITE = 0x7060,
  CLIENT_PARTY_DISMISS = 0x7061,
  CLIENT_PARTY_KICK = 0x7063,
  CLIENT_ANIMATION_INVITE = 0x3080, // CLIENT_ANIMATION_INVITE, SERVER_ANIMATION_INVITE
  kClientAgentAlchemyElixirRequest = 0x7150,
  kClientAgentAlchemyStoneRequest = 0x7151,
  CLIENT_TRANSPORT_HOME = 0x70CB, // CLIENT_TRANSPORT_HOME, CLIENT_TRANSPORT_DELETE
  CLIENT_TRANSPORT_DELETE = 0x70CB, // CLIENT_TRANSPORT_HOME, CLIENT_TRANSPORT_DELETE
  kClientAgentInventoryStorageOpenRequest = 0x703C,
  kClientAgentInventoryRepairRequest = 0x703E,
  kClientAgentCharacterUpdateBodyStateRequest = 0x70A7,

  

  SERVER_INFO = 0x2001, // LOGIN_CLIENT_INFO, LOGIN_SERVER_INFO, CLIENT_INFO, SERVER_INFO
  SERVER_HANDSHAKE = 0x5000, // LOGIN_SERVER_HANDSHAKE, SERVER_HANDSHAKE
  SERVER_PATCH_INFO = 0x600D, // LOGIN_SERVER_PATCH_INFO, LOGIN_SERVER_LAUNCHER, SERVER_PATCH_INFO
  kServerAgentAuthResponse = 0xA103,

  kServerAgentCharacterSelectionActionResponse = 0xB007,
  kServerAgentCharacterSelectionJoinResponse = 0xB001,
  SERVER_AGENT_CHARACTER_INFO_BEGIN = 0x34A5,
  kServerAgentCharacterData = 0x3013,
  SERVER_AGENT_CHARACTER_INFO_END = 0x34A6,
  kServerAgentEnvironmentCelestialPosition = 0x3020,

  kServerAgentEntitySpawn = 0x3015,
  kServerAgentEntityDespawn = 0x3016,

  kServerAgentEntityGroupspawnBegin = 0x3017,
  kServerAgentEntityGroupspawnData = 0x3019,
  kServerAgentEntityGroupspawnEnd = 0x3018,

  SERVER_ITEM_EQUIP = 0x3038,
  SERVER_ITEM_UNEQUIP = 0x3039,
  kServerAgentInventoryOperationResponse = 0xB034,
  SERVER_ANIMATION_ITEM_PICKUP = 0x3036,
  kServerAgentInventoryItemUseResponse = 0xB04C,
  SERVER_ANIMATION_ITEM_USE = 0x305C,
  SERVER_ANIMATION_CAPE = 0x3041,
  kServerAgentInventoryUpdateItem = 0x3040,

  SERVER_QUIT_GAME = 0x300A,
  SERVER_COUNTDOWN = 0xB005,
  SERVER_COUNTDOWN_INTERRUPT = 0xB006,

  kServerAgentEntityUpdatePoints = 0x304E,
  kServerAgentCharacterUpdateStats = 0x303D,
  kServerAgentCharacterIncreaseStrResponse = 0xB050,
  kServerAgentCharacterIncreaseIntResponse = 0xB051,
  kServerAgentEntitySyncPosition = 0x3028,
  kServerAgentEntityUpdateState = 0x30BF,
  kServerAgentEntityUpdateHwanLevel = 0x30DF,
  kServerAgentEntityUpdateMoveSpeed = 0x30D0,
  kServerAgentEntityRemoveOwnership = 0x304D,
  kServerAgentEntityUpdateStatus = 0x3057,
  kServerAgentEntityDamageEffect = 0x3058,
  SERVER_ANIMATION_LEVEL_UP = 0x3054,
  kServerAgentEntityUpdateExperience = 0x3056,
  kServerAgentSkillMasteryLearnResponse = 0xB0A2,
  kServerAgentSkillLearnResponse = 0xB0A1,

  kServerAgentChatUpdate = 0x3026,
  SERVER_CHAT_ACCEPT = 0xB025,

  kServerAgentActionDeselectResponse = 0xB04B,
  kServerAgentActionSelectResponse = 0xB045,
  kServerAgentActionTalkResponse = 0xB046,
  kServerAgentEntityUpdateMovement = 0xB021,
  SERVER_UNIQUE = 0x300C,

  kServerAgentCosData = 0x30C8,
  SERVER_COS_SIT_UP = 0xB0CB,
  SERVER_ANIMATION_COS_REMOVE_MENU = 0x30C9,
  SERVER_COS_DELETE = 0xB0C6,

  //Environment
  kServerEnvironmentWeather = 0x3809,

  kServerAgentSkillBegin = 0xB070,
  kServerAgentSkillEnd = 0xB071,
  kServerAgentActionCommandResponse = 0xB074,

  kServerAgentBuffAdd = 0xB0BD,
  kServerAgentBuffLink = 0xB0BE,
  kServerAgentBuffRemove = 0xB072,

  kServerAgentResurrectOption = 0x3011,
  kServerAgentAbnormalInfo = 0x30D2,

  SERVER_PARTY_FORM = 0xB069,
  SERVER_PARTY_EDIT = 0xB06A,
  SERVER_PARTY_DELETE = 0xB06B,
  SERVER_PARTY_MATCHING = 0xB06C,
  SERVER_PARTY_ACCEPT = 0xB06D,
  SERVER_PARTY_REQUEST = 0x706D,
  SERVER_PARTY_NEW_PARTY = 0x3065,
  SERVER_PARTY_CHANGES = 0x3864,
  SERVER_PARTY_INVITE = 0xB060,
  SERVER_ANIMATION_INVITE = 0x3080, // CLIENT_ANIMATION_INVITE, SERVER_ANIMATION_INVITE

  SERVER_SILK_AMOUNT = 0x3153,

  SERVER_TELEPORT = 0xB05A,
  kServerAgentGameReset = 0x34B5,

  kServerAgentInventoryStorageBegin = 0x3047,
  kServerAgentInventoryStorageData = 0x3049,
  kServerAgentInventoryStorageEnd = 0x3048,

  kServerAgentGuildStorageBegin = 0x3253,
  kServerAgentGuildStorageData = 0x3255,
  kServerAgentGuildStorageEnd = 0x3254,

  kServerAgentAlchemyElixirResponse = 0xB150,
  kServerAgentAlchemyStoneResponse = 0xB151,

  kServerAgentInventoryRepairResponse = 0xB03E,
  kServerAgentInventoryUpdateDurability = 0x3052,

  kServerAgentEntityUpdatePosition = 0xB023,
  kServerAgentEntityUpdateAngle = 0xB024,

  kClientAgentFreePvpUpdateRequest = 0x7516,
  kServerAgentFreePvpUpdateResponse = 0xB516,
  kClientAgentOperatorRequest = 0x7010,
  kServerAgentOperatorResponse = 0xB010,
};

std::string toString(Opcode opcode);

} // namespace packet

#endif // PACKET_OPCODE_HPP_