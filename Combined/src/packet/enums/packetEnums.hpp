#ifndef PACKET_ENUMS_HPP
#define PACKET_ENUMS_HPP

#include <cstdint>

namespace packet::enums {

enum class AngleAction { kObsolete=0, kGoForward=1 };

enum class CharacterSelectionAction {
  kCreate = 1,
  kList = 2,
  kDelete = 3,
  kCheckName = 4,
  kRestore = 5
};

enum class ChatType {
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

enum class LoginResult {
  kSuccess = 1,
  kFailed = 2,
  kOther = 3
};

enum class LoginBlockType {
  kPunishment = 1,
  kAccountInspection = 2,
  kNoAccountInfo = 3,
  kFreeServiceOver = 4
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

enum class VitalInfoFlag : uint8_t
{
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
  kCharacterDead = 0x1889
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

enum class ActionType : uint8_t {
  kAttack = 1,
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
  kAttack = 1,
  kTeleport = 2,
  kSprint = 8
};

enum class HitResult : uint8_t {
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

} // namespace packet::enums

#endif // PACKET_ENUMS_HPP