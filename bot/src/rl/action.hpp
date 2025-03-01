#ifndef RL_ACTION_HPP_
#define RL_ACTION_HPP_

#include "broker/eventBroker.hpp"
#include "state/machine/stateMachine.hpp"

#include <silkroad_lib/scalar_types.hpp>

#include <optional>

class Bot;

namespace rl {

// Currently, I envision the following types of actions:
//  1. Sleep for a fixed amount of time
//  2. Use a common attack (has a target)
//  3. Cancel the current action
//  4. Use a buff skill (has no target)
//  5. Use an attack skill (has a target)
//  6. Use an item

// Actions are state machines!
// TODO: For actions which send a packet, should we have a timeout if there is no response?
class Action : public state::machine::StateMachine {
public:
  Action(Bot &bot) : StateMachine(bot) {}
  virtual ~Action() = 0;
};

class Sleep : public Action {
public:
  Sleep(Bot &bot) : Action(bot) {}
  state::machine::Status onUpdate(const event::Event *event) override;
private:
  static constexpr int kSleepDurationMs{200};
  std::optional<broker::EventBroker::EventId> eventId_;
};

class CommonAttack : public Action {
public:
  CommonAttack(Bot &bot, sro::scalar_types::EntityGlobalId targetGlobalId) : Action(bot), targetGlobalId_(targetGlobalId) {}
  state::machine::Status onUpdate(const event::Event *event) override;
private:
  const sro::scalar_types::EntityGlobalId targetGlobalId_;
  bool sentPacket_{false};
};

class CancelAction : public Action {
public:
  CancelAction(Bot &bot) : Action(bot) {}
  state::machine::Status onUpdate(const event::Event *event) override;
private:
  bool sentPacket_{false};
};

class TargetlessSkill : public Action {
public:
  TargetlessSkill(Bot &bot, sro::scalar_types::ReferenceSkillId skillRefId) : Action(bot), skillRefId_(skillRefId) {}
  state::machine::Status onUpdate(const event::Event *event) override;
private:
  const sro::scalar_types::ReferenceSkillId skillRefId_;
  bool sentPacket_{false};
};

class TargetedSkill : public Action {
public:
  TargetedSkill(Bot &bot, sro::scalar_types::ReferenceSkillId skillRefId, sro::scalar_types::EntityGlobalId targetGlobalId) : Action(bot), skillRefId_(skillRefId), targetGlobalId_(targetGlobalId) {}
  state::machine::Status onUpdate(const event::Event *event) override;
private:
  const sro::scalar_types::ReferenceSkillId skillRefId_;
  const sro::scalar_types::EntityGlobalId targetGlobalId_;
  bool sentPacket_{false};
};

class UseItem : public Action {
public:
  UseItem(Bot &bot, sro::scalar_types::ReferenceObjectId itemRefId) : Action(bot), itemRefId_(itemRefId) {}
  state::machine::Status onUpdate(const event::Event *event) override;
private:
  const sro::scalar_types::ReferenceObjectId itemRefId_;
  bool sentPacket_{false};
};

} // namespace rl

#endif // RL_ACTION_HPP_