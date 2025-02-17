#ifndef STATE_MACHINE_EXECUTE_GM_COMMAND_HPP_
#define STATE_MACHINE_EXECUTE_GM_COMMAND_HPP_

#include "event/event.hpp"
#include "shared/silkroad_security.h"
#include "stateMachine.hpp"

#include <string>

namespace state::machine {

class ExecuteGmCommand : public StateMachine {
public:
  ExecuteGmCommand(Bot &bot, packet::enums::OperatorCommand gmCommand, PacketContainer gmCommandPacket);
  ~ExecuteGmCommand() override;
  Status onUpdate(const event::Event *event) override;
private:
  static inline std::string kName{"ExecuteGmCommand"};
  const packet::enums::OperatorCommand gmCommand_;
  const PacketContainer gmCommandPacket_;
  bool waitingForResponse_{false};
};

} // namespace state::machine

#endif // STATE_MACHINE_EXECUTE_GM_COMMAND_HPP_
