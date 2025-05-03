#ifndef STATE_MACHINE_MOVE_ITEM_IN_INVENTORY_HPP_
#define STATE_MACHINE_MOVE_ITEM_IN_INVENTORY_HPP_

#include "stateMachine.hpp"

#include <silkroad_lib/storage.hpp>

#include <cstdint>

namespace state::machine {

class MoveItem : public StateMachine {
public:
  MoveItem(StateMachine *parent, sro::storage::Position source, sro::storage::Position destination);
  ~MoveItem() override;
  Status onUpdate(const event::Event *event) override;
private:
  static inline std::string kName{"MoveItem"};
  const sro::storage::Position source_;
  const sro::storage::Position destination_;
  bool initialized_{false};
  bool waitingForItemToMove_{false};
};

} // namespace state::machine

#endif // STATE_MACHINE_MOVE_ITEM_IN_INVENTORY_HPP_