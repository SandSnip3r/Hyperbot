#include "stateMachine.hpp"

#include <absl/debugging/internal/demangle.h>
#include <typeinfo>

#include "bot.hpp"
namespace state::machine {

StateMachine::StateMachine(Bot &bot) : bot_(bot) {}

StateMachine::StateMachine(StateMachine *parent) : bot_(parent->bot_), parent_(parent) {}

StateMachine::~StateMachine() {
  // Undo all blocked opcodes
  for (const auto opcode : blockedOpcodes_) {
    bot_.proxy().unblockOpcode(opcode);
  }
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

Status StateMachine::onUpdateChild(const event::Event *event) {
  if (!childState_) {
    throw std::runtime_error("Cannot run onUpdate for child state machine because we do not have a child state machine");
  }
  return childState_->onUpdate(event);
}

bool StateMachine::haveChild() const {
  return static_cast<bool>(childState_);
}

void StateMachine::setChild(std::unique_ptr<StateMachine> &&newChildStateMachine) {
  childState_.reset();
  if (newChildStateMachine == nullptr) {
    throw std::runtime_error("Cannot set a nullptr child state machine");
  }
  childState_ = std::move(newChildStateMachine);
  bot_.sendActiveStateMachine();
}

void StateMachine::resetChild() {
  childState_.reset();
  bot_.sendActiveStateMachine();
}

SequentialStateMachines& StateMachine::getChildAsSequentialStateMachines() {
  if (!haveChild()) {
    throw std::runtime_error("Cannot get child as SequentialStateMachines because there is no child state machine.");
  }
  SequentialStateMachines *sequentialStateMachines = dynamic_cast<SequentialStateMachines*>(childState_.get());
  if (sequentialStateMachines == nullptr) {
    throw std::runtime_error("Cannot get child as SequentialStateMachines because the child state machine is not a SequentialStateMachines.");
  }
  return *sequentialStateMachines;
}

std::string StateMachine::activeStateMachineName() const {
  if (childState_) {
    return childState_->activeStateMachineName();
  }
  return absl::debugging_internal::DemangleString(typeid(*this).name());
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