#ifndef STATE_MACHINE_STATE_MACHINE_HPP_
#define STATE_MACHINE_STATE_MACHINE_HPP_

#include "broker/eventBroker.hpp"
#include "packet/opcode.hpp"
#include "shared/silkroad_security.h"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#define CHAR_LOG(severity) \
  LOG(severity) << characterNameForLog() << " "

#define CHAR_VLOG(verbosity) \
  VLOG(verbosity) << characterNameForLog() << " "

// Forward declarations
class Bot;
namespace event {
struct Event;
} // namespace event

namespace state::machine {

enum class Npc { kStorage, kPotion, kGrocery, kBlacksmith, kProtector, kStable };

enum class Status { kDone, kNotDone };

class SequentialStateMachines;

class StateMachine {
public:
  StateMachine(Bot &bot);
  StateMachine(StateMachine *parent);
  virtual ~StateMachine();

  // When this is called, `Bot` will have already processed the event.
  virtual Status onUpdate(const event::Event *event) = 0;

  virtual std::string activeStateMachineName() const;
protected:
  Bot &bot_;
  void pushBlockedOpcode(packet::Opcode opcode);
  virtual void injectPacket(const PacketContainer &packet, PacketContainer::Direction direction);

  std::string characterNameForLog() const;

  bool canMove() const;

  // ============ Interact with child state machine ============
  Status onUpdateChild(const event::Event *event);
  bool haveChild() const;
  void setChild(std::unique_ptr<StateMachine> &&newChildStateMachine);
  void resetChild();
  SequentialStateMachines& getChildAsSequentialStateMachines();

  template<typename StateMachineType>
  bool childIsType() const {
    // Since we already add a const below in the dynamic_cast, the user does not need to add a const. If this assert were not here, the redundant const would simply be ignored. I have added this assert to help the user maintain consistent code. Having mixed call sites with and without const throughout the codebase would be a bit confusing for a newcomer.
    static_assert(!std::is_const<StateMachineType>::value, "Qualifying with const is not necessary. Please use childIsType<Type>() instead.");
    if (!haveChild()) {
      throw std::runtime_error("Cannot check child state machine type because we do not have a child state machine");
    }
    return dynamic_cast<const StateMachineType*>(childState_.get()) != nullptr;
  }

  template<typename StateMachineType, typename... Args>
  void setChild(Args&&... args) {
    auto child = std::unique_ptr<StateMachineType>(
        new StateMachineType(this, std::forward<Args>(args)...));
    setChild(std::move(child));
  }

  // template<typename StateMachineType, typename... Args>
  // void emplaceSequentialChild(Args&&... args) {
  //   if (!childIsType<SequentialStateMachines>()) {
  //     throw std::runtime_error("Cannot emplace a sequential child state machine because the current child state is not a SequentialStateMachine.");
  //   }
  //   SequentialStateMachines &sequentialStateMachines = dynamic_cast<SequentialStateMachines&>(*childState_);
  //   sequentialStateMachines.emplace<StateMachineType>(std::forward<Args>(args)...);
  // }

  // ===========================================================

  private:
  StateMachine *parent_{nullptr};
  std::unique_ptr<StateMachine> childState_;
  std::vector<packet::Opcode> blockedOpcodes_;
};

std::ostream& operator<<(std::ostream &stream, Npc npc);

} // namespace state::machine

std::string_view toString(state::machine::Status status);

#endif // STATE_MACHINE_STATE_MACHINE_HPP_