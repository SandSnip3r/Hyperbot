#ifndef PACKET_ENUMS_HPP
#define PACKET_ENUMS_HPP

namespace packet_enums {

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

} // namespace packet_enums

#endif // PACKET_ENUMS_HPP