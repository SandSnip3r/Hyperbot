#include "gmCommandSpawnAndPickItems.hpp"

#include "bot.hpp"
#include "entity/item.hpp"
#include "event/event.hpp"
#include "packet/building/clientAgentOperatorRequest.hpp"
#include "state/machine/pickItem.hpp"
#include "state/machine/walking.hpp"

#include <silkroad_lib/position_math.hpp>

#include <absl/log/log.h>
#include <absl/strings/str_join.h>

namespace state::machine {

GmCommandSpawnAndPickItems::GmCommandSpawnAndPickItems(Bot &bot, const std::vector<Bot::ItemRequirement> &items) : StateMachine(bot), items_(items) {
  VLOG(1) << "Constructed state machine to spawn&pick: [" << absl::StrJoin(items, ", ", [this](std::string *out, const Bot::ItemRequirement &item) { absl::StrAppend(out, bot_.gameData().getItemName(item.refId), " x", item.count); }) << "]";
}

GmCommandSpawnAndPickItems::~GmCommandSpawnAndPickItems() {
}

Status GmCommandSpawnAndPickItems::onUpdate(const event::Event *event) {
  if (!initialized_) {
    // Save our original position so that we can move back to it after picking up items.
    originalPosition_ = bot_.selfState()->position();
    initialized_ = true;
  }

  if (childState_) {
    const Status status = childState_->onUpdate(event);
    if (status != Status::kDone) {
      // Do not do anything else while child state machine is active.
      return Status::kNotDone;
    }
    // Child state machine is done.
    const bool childStateWasPick = dynamic_cast<PickItem*>(childState_.get()) != nullptr;
    childState_.reset();
    if (childStateWasPick) {
      VLOG(1) << "Finished picking item";
      // If we're not at our original position, move back to it.
      if (bot_.selfState()->position() != originalPosition_) {
        VLOG(1) << "  Moving back to original position";
        setChildStateMachine<Walking>(std::vector<packet::building::NetworkReadyPosition>{originalPosition_});
        return onUpdate(event);
      }
    }
  }

  // TODO: We should take note if our GM command was successful or not.
  if (waitingForItemToSpawn_) {
    // Measured [7.99993, 20.0001]
    constexpr double closestSpawnDistance = 8;
    constexpr double farthestSpawnDistance = 20;
    constexpr double errorTolerance = 0.1;

    if (event == nullptr) {
      // No event, nothing to do.
      return Status::kNotDone;
    }
    if (event->eventCode == event::EventCode::kEntitySpawned) {
      const event::EntitySpawned& castedEvent = dynamic_cast<const event::EntitySpawned&>(*event);
      auto itemEntity = bot_.entityTracker().getEntity<entity::Item>(castedEvent.globalId);
      // Make sure spawned entity is an item.
      if (itemEntity) {
        // Make sure spawned item is the same as what we're trying to spawn.
        if (items_.empty()) {
          throw std::runtime_error("Should not be possible to have no items yet");
        }
        if (itemEntity->refObjId == items_.front().refId) {
          const double distance = sro::position_math::calculateDistance2d(bot_.selfState()->position(), itemEntity->position());
          // Make sure the spawned item is close enough to us, otherwise, it might be spawned by someone else for someone else.
          if (distance >= closestSpawnDistance-errorTolerance && distance <= farthestSpawnDistance+errorTolerance) {
            VLOG(1) << "Item we asked for spawned. Creating state machine to pick item " << bot_.gameData().getItemName(itemEntity->refObjId);
            setChildStateMachine<PickItem>(itemEntity->globalId);
            waitingForItemToSpawn_ = false;
            return onUpdate(event);
          }
        }
      }
      // Is not the item we care about. Not done yet
      return Status::kNotDone;
    } else if (event->eventCode == event::EventCode::kOperatorRequestError) {
      const auto &castedEvent = dynamic_cast<const event::OperatorRequestError&>(*event);
      if (castedEvent.operatorCommand == packet::enums::OperatorCommand::kMakeItem) {
        // Failed to spawn item.
        VLOG(1) << "GM Command to spawn item failed";
        waitingForItemToSpawn_ = false;
        // Fall through and retry.
      }
    }
  }

  if (waitingForItemToSpawn_) {
    // We're waiting for an item to spawn. Don't spawn another item yet.
    return Status::kNotDone;
  }

  return spawnNextItem();
}

Status GmCommandSpawnAndPickItems::spawnNextItem() {
  if (items_.empty()) {
    VLOG(1) << "No more items to spawn";
    // No more items to spawn.
    return Status::kDone;
  }
  // Figure out how many of this item we need to spawn, based on how many we have and how many total we need.
  const sro::pk2::ref::ItemId refId = items_.front().refId;
  uint16_t currentCount=0;
  for (const storage::Item& item : bot_.inventory()) {
    if (item.refItemId == refId) {
      currentCount += item.getQuantity();
    }
  }
  VLOG(1) << "Have " << currentCount << "/" << items_.front().count << " of " << bot_.gameData().getItemName(refId);
  if (currentCount >= items_.front().count) {
    // We have enough of this item. Move on to the next item.
    items_.erase(items_.begin());
    VLOG(1) << "Done with item " << bot_.gameData().getItemName(refId) << ". " << items_.size() << " item types remaining";
    return spawnNextItem();
  }

  // We need to spawn more of this item.
  const uint16_t countToSpawn = std::min<uint16_t>(bot_.gameData().itemData().getItemById(refId).maxStack, std::min<uint16_t>(255, items_.front().count - currentCount));
  const PacketContainer packet = packet::building::ClientAgentOperatorRequest::makeItem(refId, countToSpawn);
  VLOG(1) << "Sending packet for GM command to spawn " << countToSpawn << " x " << bot_.gameData().getItemName(refId);
  bot_.packetBroker().injectPacket(packet, PacketContainer::Direction::kClientToServer);
  waitingForItemToSpawn_ = true;
  return Status::kNotDone;
}

} // namespace state::machine