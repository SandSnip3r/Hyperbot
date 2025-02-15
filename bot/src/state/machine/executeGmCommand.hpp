#ifndef STATE_MACHINE_EXECUTE_GM_COMMAND_HPP_
#define STATE_MACHINE_EXECUTE_GM_COMMAND_HPP_

#include "event/event.hpp"
#include "shared/silkroad_security.h"
#include "stateMachine.hpp"

#include <string>

namespace state::machine {

class ExecuteGmCommand : public StateMachine {
public:
  static ExecuteGmCommand makeItem(Bot &bot, sro::scalar_types::ReferenceObjectId refItemId, uint8_t optLevelOrAmount);

  ~ExecuteGmCommand() override;
  Status onUpdate(const event::Event *event) override;
private:
  ExecuteGmCommand(Bot &bot, packet::enums::OperatorCommand gmCommand, PacketContainer gmCommandPacket);
  static inline std::string kName{"ExecuteGmCommand"};
  const packet::enums::OperatorCommand gmCommand_;
  const PacketContainer gmCommandPacket_;
};

} // namespace state::machine

#endif // STATE_MACHINE_EXECUTE_GM_COMMAND_HPP_
