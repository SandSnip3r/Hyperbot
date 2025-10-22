#include "useItem.hpp"
#include "useReturnScroll.hpp"

#include "bot.hpp"
#include "event/event.hpp"
#include "type_id/categories.hpp"

#include <stdexcept>

namespace state::machine {

UseReturnScroll::UseReturnScroll(StateMachine *parent, sro::scalar_types::StorageIndexType inventoryIndex) : StateMachine(parent), inventoryIndex_(inventoryIndex) {
  // Do some quick checks to make sure we have this item and can use it
  const auto &inventory = bot_.selfState()->inventory;
  if (!inventory.hasItem(inventoryIndex_)) {
    throw std::runtime_error("Trying use nonexistent item in inventory");
  }
  const auto *item = inventory.getItem(inventoryIndex_);
  const auto *itemAsExpendable = dynamic_cast<const storage::ItemExpendable*>(item);
  if (itemAsExpendable == nullptr) {
    throw std::runtime_error("Item is not an expendable");
  }
  if (!type_id::categories::kReturnScroll.contains(itemAsExpendable->typeId())) {
    throw std::runtime_error("Item is not a return scroll");
  }
  setChild<UseItem>(inventoryIndex_);
}

UseReturnScroll::~UseReturnScroll() {}

Status UseReturnScroll::onUpdate(const event::Event *event) {
  if (haveChild()) {
    // Have a child state, it takes priority
    const Status status = onUpdateChild(event);
    if (status == Status::kDone) {
      resetChild();
    } else {
      // Dont execute anything else in this function until the child state is done
      return Status::kNotDone;
    }
  }

  // Now, we must wait until we teleport or die
  if (event != nullptr) {
    if (event->eventCode == event::EventCode::kSelfSpawned) {
      // We just teleported
      return Status::kDone;
    }
    // TODO: Handle when we die
  }
  return Status::kNotDone;
}

} // namespace state::machine