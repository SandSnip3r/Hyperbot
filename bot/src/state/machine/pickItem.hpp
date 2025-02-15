#ifndef STATE_MACHINE_PICK_ITEM_HPP_
#define STATE_MACHINE_PICK_ITEM_HPP_

#include "stateMachine.hpp"

#include <silkroad_lib/scalar_types.hpp>

namespace state::machine {

// Pick an item off the ground. Assumes that we're already within range of the item.
class PickItem : public StateMachine {
public:
  PickItem(Bot &bot, sro::scalar_types::EntityGlobalId targetGlobalId);
  ~PickItem() override;
  Status onUpdate(const event::Event *event) override;
private:
  static inline std::string kName{"PickItem"};
  const sro::scalar_types::EntityGlobalId targetGlobalId_;
  bool waitingForItemToBePicked_{false};
};

} // namespace state::machine

#endif // STATE_MACHINE_PICK_ITEM_HPP_