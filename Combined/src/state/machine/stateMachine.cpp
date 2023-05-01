#include "stateMachine.hpp"

#include "bot.hpp"
namespace state::machine {

StateMachine::StateMachine(Bot &bot) : bot_(bot) {
}

StateMachine::~StateMachine() {
  // Undo all blocked opcodes
  for (const auto opcode : blockedOpcodes_) {
    bot_.proxy().unblockOpcode(opcode);
  }
}

void StateMachine::pushBlockedOpcode(packet::Opcode opcode) {
  if (!bot_.proxy().blockingOpcode(opcode)) {
    bot_.proxy().blockOpcode(opcode);
    blockedOpcodes_.push_back(opcode);
  }
}

void StateMachine::stateMachineCreated(const std::string &name) {
  bot_.eventBroker().publishEvent<event::StateMachineCreated>(name);
}

void StateMachine::stateMachineDestroyed() {
  bot_.eventBroker().publishEvent(event::EventCode::kStateMachineDestroyed);
}

bool StateMachine::canMove() const {
  return !(bot_.selfState().stunnedFromKnockback || bot_.selfState().stunnedFromKnockdown);
}

void StateMachine::setChildStateMachine(std::unique_ptr<StateMachine> &&newChildStateMachine) {
  childState_.reset();
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