#ifndef STATE_MACHINE_USE_ITEM_HPP_
#define STATE_MACHINE_USE_ITEM_HPP_

#include "broker/eventBroker.hpp"
#include "stateMachine.hpp"

#include "type_id/typeCategory.hpp"

#include <silkroad_lib/pk2/ref/item.hpp>
#include <silkroad_lib/scalar_types.hpp>

#include <optional>
#include <string>

namespace state::machine {

class UseItem : public StateMachine {
public:
  UseItem(Bot &bot, sro::scalar_types::StorageIndexType inventoryIndex);
  UseItem(Bot &bot, sro::pk2::ref::ItemId itemId);
  ~UseItem() override;
  Status onUpdate(const event::Event *event) override;
private:
  static inline std::string kName{"UseItem"};
  bool initialized_{false};
  std::optional<sro::scalar_types::StorageIndexType> inventoryIndex_;
  std::optional<sro::pk2::ref::ItemId> itemId_;
  type_id::TypeId itemTypeId_;
  uint16_t lastKnownQuantity_;
  std::string itemName_;

  // Item use timeout tracking
  std::optional<broker::EventBroker::EventId> itemUseTimeoutEventId_;
  static constexpr const int kItemUseTimeoutMs{666};

  void initialize();
};

} // namespace state::machine

#endif // STATE_MACHINE_USE_ITEM_HPP_