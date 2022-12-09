#ifndef STATE_MACHINE_STATEMACHINE_HPP_
#define STATE_MACHINE_STATEMACHINE_HPP_

#include "packet/opcode.hpp"

#include <string>
#include <vector>

// Forward declarations
class Bot;
namespace event {
struct Event;
} // namespace event

namespace state::machine {

enum class Npc { kStorage, kPotion, kGrocery, kBlacksmith, kProtector, kStable };

class StateMachine {
public:
  StateMachine(Bot &bot);
  virtual ~StateMachine();
  virtual void onUpdate(const event::Event *event) = 0;
  virtual bool done() const = 0; // TODO: rename to isDone
protected:
  Bot &bot_;
  void pushBlockedOpcode(packet::Opcode opcode);
  void stateMachineCreated(const std::string &name);
  void stateMachineDestroyed();
private:
  std::vector<packet::Opcode> blockedOpcodes_;
};

std::ostream& operator<<(std::ostream &stream, Npc npc);

} // namespace state::machine

#endif // STATE_MACHINE_STATEMACHINE_HPP_