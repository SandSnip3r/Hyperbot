#include "stateMachine.hpp"

#include "bot.hpp"
namespace state::machine {

StateMachine::StateMachine(Bot &bot) : bot_(bot) {}

StateMachine::StateMachine(StateMachine *parent) : bot_(parent->bot_), parent_(parent) {}

StateMachine::~StateMachine() {
  // Undo all blocked opcodes
  for (const auto opcode : blockedOpcodes_) {
    bot_.proxy().unblockOpcode(opcode);
  }
  if (destructionPromise_.has_value()) {
    destructionPromise_->set_value();
  }
}

std::future<void> StateMachine::getDestructionFuture() {
  if (!destructionPromise_.has_value()) {
    destructionPromise_.emplace();
  }
  return destructionPromise_->get_future();
}

void StateMachine::pushBlockedOpcode(packet::Opcode opcode) {
  if (!bot_.proxy().blockingOpcode(opcode)) {
    VLOG(1) << "Pushing blocked opcode " << packet::toString(opcode);
    bot_.proxy().blockOpcode(opcode);
    blockedOpcodes_.push_back(opcode);
  }
}

void StateMachine::injectPacket(const PacketContainer &packet, PacketContainer::Direction direction) {
  if (parent_ != nullptr) {
    VLOG(1) << "Delegating packet injection to parent state machine";
    parent_->injectPacket(packet, direction);
  } else {
    // No parent, inject the packet ourselves.
    bot_.injectPacket(packet, direction);
  }
}

std::string StateMachine::characterNameForLog() const {
  if (bot_.selfState() == nullptr) {
    return absl::StrFormat("[NOT_LOGGED_IN]");
  } else {
    return absl::StrFormat("[%s]", bot_.selfState()->name);
  }
}

bool StateMachine::canMove() const {
  return !(bot_.selfState()->stunnedFromKnockback || bot_.selfState()->stunnedFromKnockdown);
}

void StateMachine::setChildStateMachine(std::unique_ptr<StateMachine> &&newChildStateMachine) {
  childState_.reset();
  if (newChildStateMachine == nullptr) {
    throw std::runtime_error("Cannot set a nullptr child state machine");
  }
  childState_ = std::move(newChildStateMachine);
}

std::ostream& operator<<(std::ostream &stream, Npc npc) {
  switch (npc) {
    case Npc::kStorage:
      stream << "Storage";
      break;
    case Npc::kPotion:
      stream << "Potion";
      break;
    case Npc::kProtector:
      stream << "Protector";
      break;
    case Npc::kGrocery:
      stream << "Grocery";
      break;
    case Npc::kBlacksmith:
      stream << "Blacksmith";
      break;
    case Npc::kStable:
      stream << "Stable";
      break;
    default:
      stream << "UNKNOWN";
      break;
  }
  return stream;
}

} // namespace state::machine

std::string_view toString(state::machine::Status status) {
  if (status == state::machine::Status::kDone) {
    return "Done";
  } else if (status == state::machine::Status::kNotDone) {
    return "Not Done";
  } else {
    throw std::runtime_error("Unknown status");
  }
}