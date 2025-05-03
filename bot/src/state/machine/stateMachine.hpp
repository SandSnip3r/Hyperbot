#ifndef STATE_MACHINE_STATE_MACHINE_HPP_
#define STATE_MACHINE_STATE_MACHINE_HPP_

#include "broker/eventBroker.hpp"
#include "packet/opcode.hpp"
#include "shared/silkroad_security.h"

#include <future>
#include <memory>
#include <optional>
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

class StateMachine {
public:
  StateMachine(Bot &bot);
  StateMachine(StateMachine *parent);
  virtual ~StateMachine();

  // When this is called, `Bot` will have already processed the event.
  virtual Status onUpdate(const event::Event *event) = 0;

  std::future<void> getDestructionFuture();
protected:
  Bot &bot_;
  void pushBlockedOpcode(packet::Opcode opcode);
  virtual void injectPacket(const PacketContainer &packet, PacketContainer::Direction direction);

  std::string characterNameForLog() const;

  bool canMove() const;

  template<typename StateMachineType, typename... Args>
  void setChildStateMachine(Args&&... args) {
    childState_.reset();
    childState_ = std::unique_ptr<StateMachineType>(new StateMachineType(this, std::forward<Args>(args)...));
  }
  void setChildStateMachine(std::unique_ptr<StateMachine> &&newChildStateMachine);
  std::unique_ptr<StateMachine> childState_;
private:
  StateMachine *parent_{nullptr};
  std::vector<packet::Opcode> blockedOpcodes_;
  std::optional<std::promise<void>> destructionPromise_;
};

std::ostream& operator<<(std::ostream &stream, Npc npc);

} // namespace state::machine

std::string_view toString(state::machine::Status status);

#endif // STATE_MACHINE_STATE_MACHINE_HPP_