#ifndef STATE_MACHINE_USE_ITEM_HPP_
#define STATE_MACHINE_USE_ITEM_HPP_

#include "stateMachine.hpp"

#include "type_id/typeCategory.hpp"

#include <silkroad_lib/scalar_types.h>

namespace state::machine {

class UseItem : public StateMachine {
public:
  UseItem(Bot &bot, sro::scalar_types::StorageIndexType inventoryIndex);
  ~UseItem() override;
  void onUpdate(const event::Event *event) override;
  bool done() const override;
private:
  static inline std::string kName{"UseItem"};
  sro::scalar_types::StorageIndexType inventoryIndex_;
  type_id::TypeId itemTypeId_;
  uint16_t lastKnownQuantity_;
  bool waitingForItemToBeUsed_{false};
  bool done_{false};
};

} // namespace state::machine

#endif // STATE_MACHINE_USE_ITEM_HPP_