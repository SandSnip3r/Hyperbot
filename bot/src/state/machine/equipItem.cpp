#include "bot.hpp"
#include "state/machine/equipItem.hpp"
#include "state/machine/moveItem.hpp"
#include "type_id/categories.hpp"

#include <silkroad_lib/game_constants.hpp>

#include <absl/log/log.h>

namespace state::machine {

EquipItem::EquipItem(StateMachine *parent, sro::scalar_types::ReferenceObjectId itemRefId) : StateMachine(parent), itemRefId_(itemRefId) {
}

EquipItem::~EquipItem() {
}

Status EquipItem::onUpdate(const event::Event *event) {
  if (!initialized_) {
    std::optional<sro::scalar_types::StorageIndexType> itemSlot = bot_.inventory().findFirstItemWithRefId(itemRefId_);
    if (!itemSlot) {
      throw std::runtime_error(absl::StrFormat("We don't have item %d to equip", itemRefId_));
    }
    itemSlot_ = *itemSlot;
    CHAR_VLOG(1) << absl::StreamFormat("Found item %s in slot %d", bot_.gameData().getItemName(itemRefId_), itemSlot_);
    initialized_ = true;
  }

  if (childState_) {
    // Delegate to child state machine
    Status childStatus = childState_->onUpdate(event);
    if (childStatus == Status::kNotDone) {
      return Status::kNotDone;
    }
    // Child state finished, we're done.
    CHAR_VLOG(2) << "Child state finished, we're done.";
    childState_.reset();
    return Status::kDone;
  }

  // Where does the item go?
  const storage::ItemEquipment *equipment = dynamic_cast<const storage::ItemEquipment*>(bot_.inventory().getItem(itemSlot_));
  if (!equipment) {
    throw std::runtime_error(absl::StrFormat("Item %d is not an equipment", itemRefId_));
  }
  if (!equipment->isOneOf({type_id::categories::kAvatarHat})) {
    throw std::runtime_error(absl::StrFormat("For now, only avatar hats are supported. \"%s\" is not an avatar hat", bot_.gameData().getItemName(itemRefId_)));
  }
  CHAR_VLOG(1) << absl::StreamFormat("Item %s is an avatar hat, moving from inventory slot %d to avatar slot %d", bot_.gameData().getItemName(itemRefId_), itemSlot_, sro::game_constants::kAvatarHatSlot);
  setChildStateMachine<state::machine::MoveItem>(sro::storage::Position(sro::storage::Storage::kInventory, itemSlot_),
                                                 sro::storage::Position(sro::storage::Storage::kAvatarInventory, sro::game_constants::kAvatarHatSlot));
  return onUpdate(event);
}

} // namespace state::machine
