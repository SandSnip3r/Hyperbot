#ifndef STATE_MACHINE_SPAWN_AND_USE_REPAIR_HAMMER_IF_NECESSARY_HPP_
#define STATE_MACHINE_SPAWN_AND_USE_REPAIR_HAMMER_IF_NECESSARY_HPP_

#include "event/event.hpp"
#include "stateMachine.hpp"

#include <silkroad_lib/scalar_types.hpp>

#include <string>

namespace state::machine {

class SpawnAndUseRepairHammerIfNecessary : public StateMachine {
public:
  SpawnAndUseRepairHammerIfNecessary(StateMachine *parent);
  ~SpawnAndUseRepairHammerIfNecessary() override;
  Status onUpdate(const event::Event *event) override;
private:
  static inline std::string kName{"SpawnAndUseRepairHammerIfNecessary"};
  sro::scalar_types::ReferenceObjectId repairHammerRefId_;
  bool initialized_{false};
  bool haveRepairHammer_{false};
};

} // namespace state::machine

#endif // STATE_MACHINE_SPAWN_AND_USE_REPAIR_HAMMER_IF_NECESSARY_HPP_
