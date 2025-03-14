#ifndef STATE_MACHINE_PICK_ITEM_HPP_
#define STATE_MACHINE_PICK_ITEM_HPP_

#include "broker/eventBroker.hpp"
#include "stateMachine.hpp"

#include <silkroad_lib/scalar_types.hpp>

#include <optional>

namespace state::machine {

// Pick an item off the ground. Assumes that we're already within range of the item.
class PickItem : public StateMachine {
public:
  PickItem(StateMachine *parent, sro::scalar_types::EntityGlobalId targetGlobalId);
  ~PickItem() override;
  Status onUpdate(const event::Event *event) override;
private:
  static inline std::string kName{"PickItem"};
  const sro::scalar_types::EntityGlobalId targetGlobalId_;
  bool initialized_{false};
  sro::pk2::ref::ItemId targetRefId_;
  // When we pick an item, first it should despawn, then it should arrive in our inventory. Track these two separately so that if something arrives in our inventory matching the RefId before the item despawns, we know that it's not the item we're trying to pick up.
  std::optional<broker::EventBroker::EventId> requestTimeoutEventId_;
  bool waitingForItemToDespawn_{true};
  bool waitingForItemToArriveInInventory_{true};
  void tryPickItem();
};

} // namespace state::machine

#endif // STATE_MACHINE_PICK_ITEM_HPP_