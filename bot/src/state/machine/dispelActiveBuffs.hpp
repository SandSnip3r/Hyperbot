#ifndef STATE_MACHINE_DISPEL_ACTIVE_BUFFS_HPP_
#define STATE_MACHINE_DISPEL_ACTIVE_BUFFS_HPP_

#include "broker/eventBroker.hpp"
#include "event/event.hpp"
#include "state/machine/stateMachine.hpp"

#include <silkroad_lib/scalar_types.hpp>

#include <optional>
#include <string>

namespace state::machine {

class DispelActiveBuffs : public StateMachine {
public:
  DispelActiveBuffs(StateMachine *parent);
  ~DispelActiveBuffs() override;
  Status onUpdate(const event::Event *event) override;
private:
  static inline std::string kName{"DispelActiveBuffs"};
  std::optional<sro::scalar_types::ReferenceSkillId> buffSkillId_;
  std::optional<broker::EventBroker::EventId> dispelTimeoutEventId_;
};

} // namespace state::machine

#endif // STATE_MACHINE_DISPEL_ACTIVE_BUFFS_HPP_
