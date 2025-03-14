#include "spawnAndUseRepairHammerIfNecessary.hpp"

#include "bot.hpp"
#include "state/machine/gmCommandSpawnAndPickItems.hpp"
#include "state/machine/useItem.hpp"
#include "type_id/categories.hpp"

#include <absl/log/log.h>

namespace state::machine {

namespace {

bool isRepairHammer(const sro::pk2::ref::Item &item) {
  // The only way I know how to differentiate the repair hammer I want is by checking if it is a cash item and if it is tradeable. There must be a better way, but this should be sufficient.
  return type_id::categories::kRepair.contains(type_id::getTypeId(item)) && item.cashItem == 1 & item.canTrade == 1;
}

} // namespace

SpawnAndUseRepairHammerIfNecessary::SpawnAndUseRepairHammerIfNecessary(StateMachine *parent) : StateMachine(parent) {
  repairHammerRefId_ = bot_.gameData().itemData().getItemId(std::bind(isRepairHammer, std::placeholders::_1));
  VLOG(1) << "Figured that repair hammer ref id is " << repairHammerRefId_;
}

SpawnAndUseRepairHammerIfNecessary::~SpawnAndUseRepairHammerIfNecessary() {
}

Status SpawnAndUseRepairHammerIfNecessary::onUpdate(const event::Event *event) {
  if (!initialized_) {
    CHAR_VLOG(1) << "Initializing";
    initialized_ = true;
    bool needToRepair = false;
    // Check if we need to repair any items
    for (const storage::Item &item : bot_.inventory()) {
      if (const auto *equipment = dynamic_cast<const storage::ItemEquipment*>(&item); equipment != nullptr) {
        CHAR_VLOG(1) << bot_.gameData().getItemName(equipment->refItemId) << " durability: " << equipment->durability << "/" << equipment->maxDurability(bot_.gameData());
        if (equipment->durability < equipment->maxDurability(bot_.gameData())) {
          needToRepair = true;
          break;
        }
      }
    }
    if (!needToRepair) {
      // Nothing to repair.
      CHAR_VLOG(1) << "Nothing to repair";
      return Status::kDone;
    }
    // We need to repair.
    CHAR_VLOG(1) << "Need to repair";

    // Check if we already have a repair hammer.
    for (const storage::Item &item : bot_.inventory()) {
      if (isRepairHammer(*item.itemInfo)) {
        CHAR_VLOG(1) << "Already have a hammer! Nice.";
        haveRepairHammer_ = true;
        break;
      }
    }
  }

  if (childState_ != nullptr) {
    const Status childStatus = childState_->onUpdate(event);
    if (childStatus == Status::kNotDone) {
      // Do not continue until our child state machine is done.
      return childStatus;
    }
    // Child state machine is done.
    const bool wasUseItem = dynamic_cast<const UseItem*>(childState_.get()) != nullptr;
    const bool wasGmSpawning = dynamic_cast<const GmCommandSpawnAndPickItems*>(childState_.get()) != nullptr;
    childState_.reset();
    if (wasGmSpawning) {
      // We finished spawning and picking up the repair hammer.
      haveRepairHammer_ = true;
    } else if (wasUseItem) {
      // We just used the repair hammer.
      return Status::kDone;
    }
  }

  // Can only get here if we need to repair or if we're called another time after reporting that we're done.
  if (!haveRepairHammer_) {
    CHAR_VLOG(1) << "Need to get a repair hammer; creating state machine to spawn it with a GM command and then pick it";
    // One hammer repairs all equipment in the inventory.
    std::vector<common::ItemRequirement> repairHammerRequirements{{repairHammerRefId_, 1}};
    setChildStateMachine<GmCommandSpawnAndPickItems>(repairHammerRequirements);
    return onUpdate(event);
  }

  // At this point, we must have a repair hammer. Check which inventory slot the hammer is at.
  CHAR_VLOG(1) << "Have repair hammer; creating state machine to use it";
  std::optional<uint8_t> hammerSlot;
  for (int slot=0; slot<bot_.inventory().size(); ++slot) {
    if (bot_.inventory().hasItem(slot)) {
      const storage::Item *item = bot_.inventory().getItem(slot);
      if (isRepairHammer(*item->itemInfo)) {
        CHAR_VLOG(1) << "Found repair hammer at slot " << slot;
        hammerSlot = slot;
        break;
      }
    }
  }
  if (!hammerSlot) {
    throw std::runtime_error("Have no repair hammer!");
  }

  // Found the hammer.
  CHAR_VLOG(1) << "Constructing state machine to use repair hammer";
  setChildStateMachine<UseItem>(*hammerSlot);
  return onUpdate(event);
}

} // namespace state::machine
