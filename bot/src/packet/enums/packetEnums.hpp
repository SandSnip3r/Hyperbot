#ifndef PACKET_ENUMS_HPP
#define PACKET_ENUMS_HPP

#include <cstdint>
#include <ostream>
#include <type_traits>
#include <string>

namespace flags {

template<typename T, typename = std::enable_if_t<std::is_enum_v<T>>>
bool isSet(const T &bitmask, const T &flag) {
  using Type = std::underlying_type_t<T>;
  const auto flagAsType = static_cast<Type>(flag);
  return (static_cast<Type>(bitmask) & flagAsType) == flagAsType;
}

}

namespace packet::enums {

enum class AngleAction { kObsolete=0, kGoForward=1 };

enum class CharacterSelectionAction : uint8_t {
  kCreate = 1,
  kList = 2,
  kDelete = 3,
  kCheckName = 4,
  kRestore = 5
};

enum class ChatType : uint8_t {
  kAll = 1,
  kPm = 2,
  kAllGm = 3,
  kParty = 4,
  kGuild = 5,
  kGlobal = 6,
  kNotice = 7,
  kStall = 9,
  kUnion = 11,
  kNpc = 13,
  kAcademy = 16,
};

enum class LoginResult : uint8_t {
  kSuccess = 1,
  kFailed = 2,
  kOther = 3
};

enum class LoginBlockType : uint8_t {
  kPunishment = 1,
  kAccountInspection = 2,
  kNoAccountInfo = 3,
  kFreeServiceOver = 4
};

enum class LoginErrorCode : uint8_t {
  // Password entry has failed %d out of %d times.
  kIncorrectUserInfo = 1,
  // See LoginBlockType.
  kBlocked = 2,
  // This user is already connected. The user may still be connected because of an error that forced the game to close. Please try again in 5 minutes.
  kUserStillConnected = 4,
  // Failed to connect to server. (C5)
  kUserShardIsOutOfService = 5,
  // The server is full, please try again later.
  kServerFull = 6,
  // Failed to connect to server. (C7)
  kUserInternalError = 7,
  // Failed to connect to server. (C8)
  kUserInvalidShard = 8,
  // Failed to connect to server. (C9)
  kCannotConnectAgent = 9,
  // Failed to connect to server. (C10)
  kServerInternalError = 10,
  // Cannot connect to the server because access to the current IP has exceeded its limit.
  kIpLimitExceeded = 11,
  // UIIO_CLIENT_START_CONTENT_FAIL_BILLING_FAILED
  kBillingFailed = 12,
  // Billing server error occurred.
  kBillingServerError = 13,
  // Only adults over the age of 18 are allowed to connect to the server.
  kUnderAgeAdultOnlyServer = 14,
  // Only users over the age of 12 are allowed to connect to the server.
  kUnderAgeTeenOnlyServer = 15,
  // Adults over the age of 18 are not allowed to connect to the Teen server.
  kOverAgeTeenOnlyServer = 16,
};

enum class UpdateFlag : uint16_t {
  kNone =          0,
  kDamage =        0x01,
  kDotDamage =     0x02,
  kConsume =       0x04,
  kReverse =       0x08,
  kRegeneration =  0x10,
  kPotion =        0x20,
  kHeal =          0x40,
  kUnknown128 =    0x80, //TODO: RESEARCH
  kAbnormalState = 0x100,
  kMurderBurn =    0x200
};

enum class VitalInfoFlag : uint8_t {
  kVitalInfoHp =       1,
  kVitalInfoMp =       2,
  kVitalInfoAbnormal = 4,
  kVitalInfoHgp =      8
};

enum class AbnormalStateFlag : uint32_t {
    kNone =         0,
    kFrozen =       0x01,
    kFrostbitten =  0x02,
    kShocked =      0x04,
    kBurnt =        0x08,
    kPoisoned =     0x10,
    kZombie =       0x20,
    //All effects below carry extra byte for level
    kSleep =        0x40,
    kBind =         0x80,
    kDull =         0x100,
    kFear =         0x200,
    kShortSighted = 0x400,
    kBleed =        0x800,
    kPetrify =      0x1000,
    kDarkness =     0x2000,
    kStunned =      0x4000,
    kDisease =      0x8000,
    kConfusion =    0x10000,
    kDecay =        0x20000,
    kWeak =         0x40000,
    kImpotent =     0x80000,
    kDivision =     0x100000,
    kPanic =        0x200000,
    kCombustion =   0x400000,
    kEmptyBit23 =   0x800000,
    kHidden =       0x1000000,
    kEmptyBit25 =   0x2000000,
    kEmptyBit26 =   0x4000000,
    kEmptyBit27 =   0x8000000,
    kEmptyBit28 =   0x10000000,
    kEmptyBit29 =   0x20000000,
    kEmptyBit30 =   0x40000000,
    kEmptyBit31 =   0x80000000
};

enum class InventoryErrorCode : uint16_t {
  // Still have time to reuse the item.
  kWaitForReuseDelay = 0x185B,
  // Cannot the use selected item while dead.
  kCharacterDead = 0x1889,
  // Encountered when we try to use the last item in the stack and it was already used
  kItemDoesNotExist = 0x1809
};

enum class ItemMovementType : uint8_t {
  kUpdateSlotsInventory =      0,
  kUpdateSlotsChest =          1,
  kChestDepositItem =          2,
  kChestWithdrawItem =         3,
  kSetExchangeItem =           4, //SP_ADD_EXCHANGE
  kCancelExchangeItem =        5, //SP_DEL_EXCHANGE
  kPickItem =                  6,
  kDropItem =                  7,
  kBuyItem =                   8,
  kSellItem =                  9,
  kDropGold =                  10,
  kChestWithdrawGold =         11,
  kChestDepositGold =          12,
  kSetExchangeGold =           13, //SP_UPDATE_EXCHANGE_GOLD
  kAddItemByServer =           14,
  kRemoveItemByServer =        15,
  kUpdateSlotsInventoryCos =   16,
  kPickItemCos =               17,
  kDropItemCos =               18,
  kBuyItemCos =                19,
  kSellItemCos =               20,
  kAddCositemByServer =        21,
  kDelCositemByServer =        22,
  // Missing                   23
  kBuyCashItem =               24,
  // Missing                   25
  kMoveItemCosToInventory =    26,
  kMoveItemInventoryToCos =    27,
  kPickItemByOther =           28,
  kUpdateSlotsGuildChest =     29,
  kGuildChestDepositItem =     30,
  kGuildChestWithdrawItem =    31,
  kGuildChestWithdrawGold =    32,
  kGuildChestDepositGold =     33,
  kBuyback =                   34,
  kMoveItemAvatarToInventory = 35,
  kMoveItemInventoryToAvatar = 36,
  // Missing                   37
  kMoveItemTradeNow =          38,
  kPushItemIntoMagicCube =     39,
  kPopItemFromMagicCube =      40,
  kDelItemInMagicCube =        41,
  kActivateMagicCube =         42,
  kBuyItemWithToken =          43,
  kPickSpecialItem =           44,
  // Missing                   45-51
  kPickSpecialItemBySilkpet =  52,
  // Missing                   53
  kPickSpecialItemByOther =    54,
};

enum class StateType : uint8_t {
  kLifeState = 0,
  kMotionState = 1,
  //2, 3, 5, 6, 9, 10
  kBodyState = 4,
  kPVPState = 7,
  kBattleState = 8,
  kScrollState = 11
};

enum class BodyState : uint8_t {
  kNormal = 0,
  kHwan = 1,
  kUntouchable = 2,
  kInvincibleGm = 3,
  kInvisibleGm = 4,
  kBerserker = 5, // Something to do with Roc
  kStealth = 6,
  kInvisible = 7
};

enum class CommandType : uint8_t {
  kExecute = 1,
  kCancel = 2
};

enum ActionState : uint8_t {
  kQueued = 1,
  kEnd = 2,
  kError = 3
};

enum class ActionType : uint8_t {
  kAttack = 1, // Common attack
  kPickup = 2,
  kTrace = 3,
  kCast = 4,
  kDispel = 5
};

enum class TargetType : uint8_t {
  kNone = 0,
  kEntity = 1,
  kLand = 2
};

enum class ActionFlag : uint8_t {
  kNone = 0,
  kAttack = 1,
  kTeleport = 2,
  kSprint = 8
};

enum class HitResult : uint8_t {
  kNone = 0,
  kBlocked = 2,
  kKnockdown = 4,
  kKnockback = 5,
  kCopy = 8, // Copies previous hit
  kKill = 128
};

enum class DamageFlag : uint8_t {
  kNormal = 1,
  kCritical = 2,
  kHwan = 4,
  // Missing 8
  kEffect = 16
};

enum class TalkOption : uint8_t {
  kStore = 1,
  kQuest = 2,
  kStorage = 3,
  kRepair = 4,

  kMonster = 5, // 圭쨍 (Click)

  kUnknown6 = 6, // 캐 (Casserole)

  /// <summary>
  /// UIIT_CTL_RECALL_POSITION
  /// <para>Designate as return/recall point</para>
  /// </summary>
  kSetReturnPoint = 7,

  /// <summary>
  /// UIIT_CTL_TELEPORT_TARGET
  /// <para>Select teleport area</para>
  /// </summary>
  kTeleportTarget = 8,

  /// <summary>
  /// UIIT_CTL_TELEPORT_TO_RESURRECT_POS
  /// <param>Teleport to return point.</param>
  /// </summary>
  kTeleportReturnPoint = 9,

  /// <summary>
  /// UIIT_MSG_CIRCULATION_WITHDRAW_SKILL
  /// <para>Withdrawing Skill</para>
  /// </summary>
  kWithdrawSkill = 10,

  /// <summary>
  /// Nothing visual
  /// </summary>
  kStable = 11,

  /// <summary>
  /// Goods & Export Details
  /// </summary>
  kTrade = 12,

  kGuild = 15,

  /// <summary>
  /// SN_TALK_CH_GACHA_MACHINE_2
  /// <para>Participate in the game.</para>
  /// </summary>
  kMagicPopPlay = 17,

  /// <summary>
  /// SN_TALK_CH_GACHA_OPERATOR_2
  /// <para>Exchange Item Exchange Coupon to Item</para>
  /// </summary>
  kMagicPopExchange = 18,

  /// <summary>
  /// UIIT_MSG_XMAS_EVENT_CHANGE
  /// <para>Give the socks and receive a gift.</para>
  /// </summary>
  kEventChristmasExchange = 19,

  kTrader = 20,
  kThief = 21,
  kHunter = 22,

  kFortressAdministration = 23,
  kFortressApplication = 24,
  kFortressStructManagement = 25,
  kFortressItemProduction = 26,
  kFortressTraining = 27,

  kTeleportLastLocation = 28,

  kTeleportGuide = 30,

  kFortressPully = 31,

  kGrantMagicOption = 32,

  kArenaManager = 33,
  kArenaItemManager = 34,

  kConsigment = 35,

  kSummonPartyMember = 39,

  kTeleportExitFortressDungeon = 40,
  kTeleportExitDungeon = 41,
};

enum class RepairType : uint8_t {
  kRepairOne = 1,
  kRepairAll = 2,
};

enum class UpdatePointsType : uint8_t {
  kGold = 1,
  kSp = 2,
  kStatPoint = 3,
  kHwan = 4,
  //8 = ?
  kAp = 16, // (Egypt)
};

enum class AcademyBuffUpdateFlag : uint8_t {
  kCumulatedSize = 0x0F, // 1
  kAccumulatedSize = 0xF0, // 16
};

enum class ItemUpdateFlag : uint8_t {
  kRefObjID = 1,
  kOptLevel = 2,
  kVariance = 4,
  kQuantity = 8,
  kDurability = 16,
  kMagParams = 32,
  kState = 64,
  kUnknown128 = 128,
};

enum class GroupSpawnType {
  kSpawn=1,
  kDespawn=2
};

enum class FreePvpMode : uint8_t {
  kNone = 0,
  kRed = 1,
  kBlack = 2,
  kBlue = 3,
  kWhite = 4,
  kYellow = 5
};

enum class OperatorCommand : uint16_t {
  kFindUser = 1,
  kGoTown = 2,
  kToTown = 3,
  kWorldStatus = 4,
  kStat = 5,
  kLoadMonster = 6,
  kMakeItem = 7,
  kMoveToUser = 8,
  //9,    /tranquilize?
  kSetTime = 10,
  kZoe = 12,
  kBan = 13,
  kInvisible = 14,
  kInvincible = 15,
  kWarpPoint = 16, // also posto
  kRecallUser = 17,
  kRecallGuild = 18,
  kInstance = 19,
  kMobKill = 20,
  //21,   /EVENTON?
  //22,   /EVENTOFF?
  //23,   /BLOCKLOGOUT?
  //24,   /ALLOWLOGOUT?
  kLieName = 25,
  kRealName = 26,
  kInitQ = 27,
  kResetQ = 28,
  kCompQ = 29,
  kRemoveQ = 30,
  kMoveToNPC = 31,
  kSiege = 33,
  kMakeRentItem = 38,
  kSpawnUniqueLoc = 42,
  kSpawnUniqueAll = 43,
  kBattleArena = 50,
  kTriggerAction = 55,
};

enum class CosCommandType : uint8_t {
  kMove = 1,
  kAttack = 2,
  kPick = 8,
  kFollow = 9,
  kCharm = 11,
};

enum class AlchemyAction : uint8_t {
  kCancel = 1,
  kFuse = 2,
  kSocketCreate = 3,
  kSocketRemove = 4
};

enum class AlchemyType : uint8_t {
  kDisjoin = 1,
  kManufacture = 2,
  kElixir = 3,
  kMagicStone = 4,
  kAttributeStone = 5,
  // 6?
  // 7?
  kAdvancedElixir = 8,
  // SocketCreate = 8, //different alchemyAction
  kSocketInsert = 9,
  kSocketRemove = 10 //different alchemyAction
};

enum class ResurrectionOptionFlag : uint8_t {
  kAtSpecifiedPoint = 1, // 1 = Ressurect at the specified point
  kAtPresentPoint = 2,   // 2 = Ressurect at the present point.
  kNormal = 4,
  // 02 = PVP / CTF
  // 04 = Normal
  // 06 = Fortress
};

std::ostream& operator<<(std::ostream &stream, const ActionState &enumVal);
std::ostream& operator<<(std::ostream &stream, const CommandType &enumVal);
std::ostream& operator<<(std::ostream &stream, const ActionType &enumVal);
std::ostream& operator<<(std::ostream &stream, const TargetType &enumVal);
std::ostream& operator<<(std::ostream &stream, const AbnormalStateFlag &enumVal);

std::string toString(AbnormalStateFlag flag);

} // namespace packet::enums

#endif // PACKET_ENUMS_HPP