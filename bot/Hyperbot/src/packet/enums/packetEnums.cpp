#include "packetEnums.hpp"

namespace packet::enums {

std::ostream& operator<<(std::ostream &stream, const ActionState &enumVal) {
  switch (enumVal) {
    case ActionState::kQueued:
      stream << "Queued";
      break;
    case ActionState::kEnd:
      stream << "End";
      break;
    case ActionState::kError:
      stream << "Error";
      break;
    default:
      stream << "UNKNOWN";
      break;
  }
  return stream;
}

std::ostream& operator<<(std::ostream &stream, const CommandType &enumVal) {
  switch (enumVal) {
    case CommandType::kExecute:
      stream << "Execute";
      break;
    case CommandType::kCancel:
      stream << "Cancel";
      break;
    default:
      stream << "UNKNOWN";
      break;
  }
  return stream;
}

std::ostream& operator<<(std::ostream &stream, const ActionType &enumVal) {
  switch (enumVal) {
    case ActionType::kAttack:
      stream << "Attack";
      break;
    case ActionType::kPickup:
      stream << "Pickup";
      break;
    case ActionType::kTrace:
      stream << "Trace";
      break;
    case ActionType::kCast:
      stream << "Cast";
      break;
    case ActionType::kDispel:
      stream << "Dispel";
      break;
    default:
      stream << "UNKNOWN";
      break;
  }
  return stream;
}

std::ostream& operator<<(std::ostream &stream, const TargetType &enumVal) {
  switch (enumVal) {
    case TargetType::kNone:
      stream << "None";
      break;
    case TargetType::kEntity:
      stream << "Entity";
      break;
    case TargetType::kLand:
      stream << "Land";
      break;
    default:
      stream << "UNKNOWN";
      break;
  }
  return stream;
}

std::ostream& operator<<(std::ostream &stream, const AbnormalStateFlag &enumVal) {
  if (enumVal == AbnormalStateFlag::kNone) {
    stream << "None";
    return stream;
  }
  bool printedOne{false};
  auto print = [&printedOne, &stream](const auto &str) {
    if (printedOne) {
      stream << ',';
    }
    stream << str;
    printedOne = true;
  };
  if (flags::isSet(enumVal, AbnormalStateFlag::kFrozen)) {
    print("Frozen");
  }
  if (flags::isSet(enumVal, AbnormalStateFlag::kFrostbitten)) {
    print("Frostbitten");
  }
  if (flags::isSet(enumVal, AbnormalStateFlag::kShocked)) {
    print("Shocked");
  }
  if (flags::isSet(enumVal, AbnormalStateFlag::kBurnt)) {
    print("Burnt");
  }
  if (flags::isSet(enumVal, AbnormalStateFlag::kPoisoned)) {
    print("Poisoned");
  }
  if (flags::isSet(enumVal, AbnormalStateFlag::kZombie)) {
    print("Zombie");
  }
  if (flags::isSet(enumVal, AbnormalStateFlag::kSleep)) {
    print("Sleep");
  }
  if (flags::isSet(enumVal, AbnormalStateFlag::kBind)) {
    print("Bind");
  }
  if (flags::isSet(enumVal, AbnormalStateFlag::kDull)) {
    print("Dull");
  }
  if (flags::isSet(enumVal, AbnormalStateFlag::kFear)) {
    print("Fear");
  }
  if (flags::isSet(enumVal, AbnormalStateFlag::kShortSighted)) {
    print("ShortSighted");
  }
  if (flags::isSet(enumVal, AbnormalStateFlag::kBleed)) {
    print("Bleed");
  }
  if (flags::isSet(enumVal, AbnormalStateFlag::kPetrify)) {
    print("Petrify");
  }
  if (flags::isSet(enumVal, AbnormalStateFlag::kDarkness)) {
    print("Darkness");
  }
  if (flags::isSet(enumVal, AbnormalStateFlag::kStunned)) {
    print("Stunned");
  }
  if (flags::isSet(enumVal, AbnormalStateFlag::kDisease)) {
    print("Disease");
  }
  if (flags::isSet(enumVal, AbnormalStateFlag::kConfusion)) {
    print("Confusion");
  }
  if (flags::isSet(enumVal, AbnormalStateFlag::kDecay)) {
    print("Decay");
  }
  if (flags::isSet(enumVal, AbnormalStateFlag::kWeak)) {
    print("Weak");
  }
  if (flags::isSet(enumVal, AbnormalStateFlag::kImpotent)) {
    print("Impotent");
  }
  if (flags::isSet(enumVal, AbnormalStateFlag::kDivision)) {
    print("Division");
  }
  if (flags::isSet(enumVal, AbnormalStateFlag::kPanic)) {
    print("Panic");
  }
  if (flags::isSet(enumVal, AbnormalStateFlag::kCombustion)) {
    print("Combustion");
  }
  if (flags::isSet(enumVal, AbnormalStateFlag::kEmptyBit23)) {
    print("EmptyBit23");
  }
  if (flags::isSet(enumVal, AbnormalStateFlag::kHidden)) {
    print("Hidden");
  }
  if (flags::isSet(enumVal, AbnormalStateFlag::kEmptyBit25)) {
    print("EmptyBit25");
  }
  if (flags::isSet(enumVal, AbnormalStateFlag::kEmptyBit26)) {
    print("EmptyBit26");
  }
  if (flags::isSet(enumVal, AbnormalStateFlag::kEmptyBit27)) {
    print("EmptyBit27");
  }
  if (flags::isSet(enumVal, AbnormalStateFlag::kEmptyBit28)) {
    print("EmptyBit28");
  }
  if (flags::isSet(enumVal, AbnormalStateFlag::kEmptyBit29)) {
    print("EmptyBit29");
  }
  if (flags::isSet(enumVal, AbnormalStateFlag::kEmptyBit30)) {
    print("EmptyBit30");
  }
  if (flags::isSet(enumVal, AbnormalStateFlag::kEmptyBit31)) {
    print("EmptyBit31");
  }
  return stream;
}

} // namespace packet::enums