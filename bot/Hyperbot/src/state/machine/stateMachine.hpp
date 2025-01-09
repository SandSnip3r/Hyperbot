#ifndef STATE_MACHINE_STATE_MACHINE_HPP_
#define STATE_MACHINE_STATE_MACHINE_HPP_

#include "broker/eventBroker.hpp"
#include "packet/opcode.hpp"

#include <memory>
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

  // When this is called, `Bot` will have already processed the event.
  virtual void onUpdate(const event::Event *event) = 0;
  virtual bool done() const = 0; // TODO: rename to isDone
protected:
  Bot &bot_;
  void pushBlockedOpcode(packet::Opcode opcode);
  void stateMachineCreated(const std::string &name);
  void stateMachineDestroyed();

  bool canMove() const;

  template<typename StateMachineType, typename... Args>
  void setChildStateMachine(Args&&... args) {
    childState_.reset();
    childState_ = std::unique_ptr<StateMachineType>(new StateMachineType(bot_, std::forward<Args>(args)...));
  }
  void setChildStateMachine(std::unique_ptr<StateMachine> &&newChildStateMachine);
  std::unique_ptr<StateMachine> childState_;
private:
  std::vector<packet::Opcode> blockedOpcodes_;
  broker::EventBroker::EventId debugEventId_;
};

std::ostream& operator<<(std::ostream &stream, Npc npc);

} // namespace state::machine

#endif // STATE_MACHINE_STATE_MACHINE_HPP_