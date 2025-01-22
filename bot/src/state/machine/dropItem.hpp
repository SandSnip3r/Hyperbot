#ifndef STATE_MACHINE_DROP_ITEM_HPP_
#define STATE_MACHINE_DROP_ITEM_HPP_

#include "stateMachine.hpp"

#include <silkroad_lib/scalar_types.hpp>

namespace state::machine {

// Drop an item.
class DropItem : public StateMachine {
public:
  DropItem(Bot &bot, sro::scalar_types::StorageIndexType inventorySlot);
  ~DropItem() override;
  void onUpdate(const event::Event *event) override;
  bool done() const override;
private:
  static inline std::string kName{"DropItem"};
  sro::scalar_types::StorageIndexType inventorySlot_;
  bool done_{false};
  sro::scalar_types::ReferenceObjectId refId_;
  bool waitingForItemToBeDropped_{false};
};

} // namespace state::machine

#endif // STATE_MACHINE_DROP_ITEM_HPP_