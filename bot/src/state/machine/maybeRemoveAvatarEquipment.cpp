#include "state/machine/maybeRemoveAvatarEquipment.hpp"
#include "state/machine/moveItem.hpp"

#include "bot.hpp"

#include <silkroad_lib/game_constants.hpp>

#include <absl/log/log.h>

namespace state::machine {

MaybeRemoveAvatarEquipment::MaybeRemoveAvatarEquipment(StateMachine *parent, sro::scalar_types::StorageIndexType avatarSlot, std::function<sro::scalar_types::StorageIndexType(Bot&)> targetSlot) : StateMachine(parent), avatarSlot_(avatarSlot), targetSlot_(targetSlot) {
}

MaybeRemoveAvatarEquipment::~MaybeRemoveAvatarEquipment() {
}

Status MaybeRemoveAvatarEquipment::onUpdate(const event::Event *event) {
  if (haveChild()) {
    // Have a child state, it takes priority
    const Status status = onUpdateChild(event);
    if (status == Status::kDone) {
      // Child state is done
      CHAR_VLOG(2) << absl::StreamFormat("Child state is done");
      resetChild();
      if (unequipping_) {
        CHAR_VLOG(2) << "Was unequipping, now maybe need to move item to target slot";
        unequipping_ = false;
        const sro::scalar_types::StorageIndexType targetSlot = targetSlot_(bot_);
        if (intermediateSlot_ == targetSlot) {
          // Unequipped directly to target slot. Done.
          return Status::kDone;
        } else {
          // Now need to do one more move to the target slot.
          CHAR_VLOG(1) << absl::StreamFormat("Moving item from intermediate slot %d to target slot %d", intermediateSlot_, targetSlot);
          setChild<MoveItem>(sro::storage::Position(sro::storage::Storage::kInventory, intermediateSlot_),
                                         sro::storage::Position(sro::storage::Storage::kInventory, targetSlot));
          return onUpdate(nullptr);
        }
      } else {
        CHAR_VLOG(2) << "Was final move, all done";
        return Status::kDone;
      }
      throw std::runtime_error("Logic check. Should not be able to get here.");
    }
    return status;
  }

  // Check if there is an item in the avatar slot.
  if (!bot_.selfState()->avatarInventory.hasItem(avatarSlot_)) {
    // No item.
    CHAR_VLOG(1) << absl::StreamFormat("No avatar item equipped in pos %d", avatarSlot_);
    return Status::kDone;
  }

  // No child state.
  // Avatar->inventory must go to the first free inventory slot. Moving the item to any other inventory slot is not possible.
  const std::optional<sro::scalar_types::StorageIndexType> firstFreeSlot = bot_.inventory().firstFreeSlot(sro::game_constants::kFirstInventorySlot);
  if (!firstFreeSlot) {
    throw std::runtime_error(absl::StrFormat("No free slot in inventory"));
  }
  intermediateSlot_ = *firstFreeSlot;
  CHAR_VLOG(1) << absl::StreamFormat("Constructing state machine to move item from avatar inventory %d to inventory %d", avatarSlot_, intermediateSlot_);
  unequipping_ = true;
  setChild<MoveItem>(sro::storage::Position(sro::storage::Storage::kAvatarInventory, avatarSlot_),
                                 sro::storage::Position(sro::storage::Storage::kInventory, intermediateSlot_));
  // TODO: Moving an item might not go to our exact target slot. We should track the item ID to be able to catch the actual destination(intermediate) slot that the item goes to so that we can then move it to the real destination slot.
  return onUpdate(nullptr);
}

} // namespace state::machine
