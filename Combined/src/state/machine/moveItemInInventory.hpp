#ifndef STATE_MACHINE_MOVE_ITEM_IN_INVENTORY_HPP_
#define STATE_MACHINE_MOVE_ITEM_IN_INVENTORY_HPP_

#include "stateMachine.hpp"

#include <cstdint>

namespace state::machine {

class MoveItemInInventory : public StateMachine {
public:
  MoveItemInInventory(Bot &bot, uint8_t srcSlot, uint8_t destSlot);
  void onUpdate(const event::Event *event) override;
  bool done() const override;
private:
  uint8_t srcSlot_;
  const uint8_t destSlot_;
  bool done_{false};
  bool waitingForItemToMove_{false};
};

} // namespace state::machine

#endif // STATE_MACHINE_MOVE_ITEM_IN_INVENTORY_HPP_