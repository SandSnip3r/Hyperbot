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

GmCommandSpawnAndPickItems::GmCommandSpawnAndPickItems(StateMachine *parent, const std::vector<common::ItemRequirement> &items) : StateMachine(parent), items_(items) {
  VLOG(1) << "Constructed state machine to spawn&pick: [" << absl::StrJoin(items, ", ", [this](std::string *out, const common::ItemRequirement &item) { absl::StrAppend(out, bot_.gameData().getItemName(item.refId), " x", item.count); }) << "]";
}

GmCommandSpawnAndPickItems::~GmCommandSpawnAndPickItems() {
  if (requestTimeoutEventId_) {
    bot_.eventBroker().cancelDelayedEvent(*requestTimeoutEventId_);
    requestTimeoutEventId_.reset();
  }
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
    const bool childStateWasWalking = dynamic_cast<Walking*>(childState_.get()) != nullptr;
    childState_.reset();
    if (childStateWasPick) {
      CHAR_VLOG(1) << "Finished picking item";
      // If we're not at our original position, move back to it.
      if (bot_.selfState()->position() != originalPosition_) {
        CHAR_VLOG(1) << "  Moving back to original position";
        setChildStateMachine<Walking>(std::vector<packet::building::NetworkReadyPosition>{originalPosition_});
        return onUpdate(event);
      }
    } else if (childStateWasWalking && tryingNudgePosition_) {
      CHAR_VLOG(1) << "Finished nudging position";
      tryingNudgePosition_ = false;
    }
  }

  if (requestTimeoutEventId_) {
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
      try {
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
              CHAR_VLOG(1) << "Item (#" << itemEntity->globalId << ") we asked for spawned. Creating state machine to pick item " << bot_.gameData().getItemName(itemEntity->refObjId);
              if (requestTimeoutEventId_) {
                bot_.eventBroker().cancelDelayedEvent(*requestTimeoutEventId_);
                requestTimeoutEventId_.reset();
              }
              setChildStateMachine<PickItem>(itemEntity->globalId);
              return onUpdate(event);
            } else {
              CHAR_VLOG(1) << absl::StreamFormat("Spawned item %d is too far (%f - us:%s, item:%s, original pos:%s) away to pick.", itemEntity->globalId, distance, bot_.selfState()->position().toString(), itemEntity->position().toString(), originalPosition_.toString());
            }
          }
        }
        // Is not the item we care about. Not done yet
      } catch (std::exception &ex) {
        // There is a chance that this thing has already despawned by the time we're processing this event.
        // That is okay. In this specific case, it is probably because our PVP opponent picked up their item.
        LOG(WARNING) << "Entity spawned, but it is not tracked: " << ex.what();
      }
      return Status::kNotDone;
    } else if (event->eventCode == event::EventCode::kOperatorRequestError) {
      const auto &castedEvent = dynamic_cast<const event::OperatorRequestError&>(*event);
      if (castedEvent.operatorCommand == packet::enums::OperatorCommand::kMakeItem) {
        // Failed to spawn item.
        CHAR_VLOG(1) << "GM Command to spawn item failed";
        if (requestTimeoutEventId_) {
          bot_.eventBroker().cancelDelayedEvent(*requestTimeoutEventId_);
          requestTimeoutEventId_.reset();
        }
        // Fall through and retry.
      }
    } else if (event->eventCode == event::EventCode::kTimeout &&
               requestTimeoutEventId_ &&
               *requestTimeoutEventId_ == event->eventId) {
      CHAR_VLOG(1) << "Timeout waiting for item to spawn. Retrying";
      requestTimeoutEventId_.reset();

      // It can happen that our position is bugged and when spawning items, we think that they are too far away.
      // If we fail too many times, we'll try to correct for this by moving a bit.
      ++timeoutCount_;
      if (timeoutCount_ >= kMaxTimeoutCount) {
        CHAR_VLOG(1) << "Timed out too many times, nudging position";
        std::vector<packet::building::NetworkReadyPosition> steps;
        steps.push_back(sro::position_math::createNewPositionWith2dOffset(originalPosition_, 30.0, 30.0));
        steps.push_back(originalPosition_);
        setChildStateMachine<Walking>(steps);
        timeoutCount_ = 0;
        tryingNudgePosition_ = true;
        return onUpdate(event);
      }
      return spawnNextItem();
    }
  }

  if (requestTimeoutEventId_) {
    // We're waiting for an item to spawn. Don't spawn another item yet.
    return Status::kNotDone;
  }

  return spawnNextItem();
}

Status GmCommandSpawnAndPickItems::spawnNextItem() {
  if (items_.empty()) {
    CHAR_VLOG(1) << "No more items to spawn";
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
  CHAR_VLOG(1) << "Have " << currentCount << "/" << items_.front().count << " of " << bot_.gameData().getItemName(refId);
  if (currentCount >= items_.front().count) {
    // We have enough of this item. Move on to the next item.
    items_.erase(items_.begin());
    CHAR_VLOG(1) << "Done with item " << bot_.gameData().getItemName(refId) << ". " << items_.size() << " item types remaining";
    return spawnNextItem();
  }

  // We need to spawn more of this item.
  const uint16_t countToSpawn = std::min<uint16_t>(bot_.gameData().itemData().getItemById(refId).maxStack, std::min<uint16_t>(255, items_.front().count - currentCount));
  const PacketContainer packet = packet::building::ClientAgentOperatorRequest::makeItem(refId, countToSpawn);
  CHAR_VLOG(1) << "Sending packet for GM command to spawn " << countToSpawn << " x " << bot_.gameData().getItemName(refId);
  injectPacket(packet, PacketContainer::Direction::kBotToServer);
  requestTimeoutEventId_ = bot_.eventBroker().publishDelayedEvent(event::EventCode::kTimeout, std::chrono::milliseconds(888));
  return Status::kNotDone;
}

} // namespace state::machine