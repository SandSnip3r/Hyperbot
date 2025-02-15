#include "gmCommandSpawnAndPickItems.hpp"

#include "bot.hpp"
#include "entity/item.hpp"
#include "event/event.hpp"
#include "packet/building/clientAgentOperatorRequest.hpp"
#include "state/machine/pickItem.hpp"
#include "state/machine/walking.hpp"

#include <silkroad_lib/position_math.hpp>

#include <absl/log/log.h>

namespace state::machine {

GmCommandSpawnAndPickItems::GmCommandSpawnAndPickItems(Bot &bot, const std::vector<ItemRequest> &items) : StateMachine(bot), items_(items) {
  stateMachineCreated(kName);
  for (const auto &item : items_) {
    VLOG(1) << "Going to use GM commands to spawn and pick " << item.quantity << " x " << bot_.gameData().getItemName(item.refItemId);
  }
  spawnNextItem();
  originalPosition_ = bot_.selfState()->position();
}

GmCommandSpawnAndPickItems::~GmCommandSpawnAndPickItems() {
  stateMachineDestroyed();
}

Status GmCommandSpawnAndPickItems::onUpdate(const event::Event *event) {
  if (event == nullptr) {
    // No event, nothing to do.
    return Status::kNotDone;
  }

  if (childState_) {
    const Status status = childState_->onUpdate(event);
    if (status == Status::kDone) {
      const bool childStateWasPick = dynamic_cast<PickItem*>(childState_.get()) != nullptr;
      const bool childStateWasWalk = dynamic_cast<Walking*>(childState_.get()) != nullptr;
      childState_.reset();
      if (childStateWasPick) {
        VLOG(1) << "Finished picking item";
        // If we're not at our original position, move back to it.
        if (bot_.selfState()->position() != originalPosition_) {
          VLOG(1) << "  Moving back to original position";
          setChildStateMachine<Walking>(std::vector<packet::building::NetworkReadyPosition>{originalPosition_});
          return onUpdate(event);
        } else {
          // Didn't move, so we're at our original position. Spawn the next item.
          VLOG(1) << "  Spawning next item";
          const Status status = spawnNextItem();
          if (status == Status::kDone) {
            return Status::kDone;
          }
        }
      } else if (childStateWasWalk) {
        VLOG(1) << "Finished walking back to center";
        // We're back at our original position. Spawn the next item.
        const Status status = spawnNextItem();
        if (status == Status::kDone) {
          return Status::kDone;
        }
      }
    }
    return Status::kNotDone;
  }

  // Measured [7.99993, 20.0001]
  constexpr double closestSpawnDistance = 8;
  constexpr double farthestSpawnDistance = 20;
  constexpr double errorTolerance = 0.1;
  // TODO: We should take note if our GM command was successful or not.
  if (waitingForItemToSpawn_ && event->eventCode == event::EventCode::kEntitySpawned) {
    const event::EntitySpawned& castedEvent = dynamic_cast<const event::EntitySpawned&>(*event);
    auto itemEntity = bot_.entityTracker().getEntity<entity::Item>(castedEvent.globalId);
    // Make sure spawned entity is an item.
    if (itemEntity) {
      // Make sure spawned item is the same as what we're trying to spawn.
      if (itemEntity->refObjId == items_.at(currentIndex_).refItemId) {
        const double distance = sro::position_math::calculateDistance2d(bot_.selfState()->position(), itemEntity->position());
        // Make sure the spawned item is close enough to us, otherwise, it might be spawned by someone else for someone else.
        if (distance >= closestSpawnDistance-errorTolerance && distance <= farthestSpawnDistance+errorTolerance) {
          VLOG(1) << "Creating state machine to pick item " << bot_.gameData().getItemName(itemEntity->refObjId);
          setChildStateMachine<PickItem>(itemEntity->globalId);
          waitingForItemToSpawn_ = false;
          return onUpdate(event);
        }
      }
    }
  }
  return Status::kNotDone;
}

Status GmCommandSpawnAndPickItems::spawnNextItem() {
  // Skip items which we need none of.
  while (currentIndex_ < items_.size() && items_.at(currentIndex_).quantity == 0) {
    ++currentIndex_;
  }

  // If there are no items left, we're done.
  if (currentIndex_ >= items_.size()) {
    return Status::kDone;
  }

  // There is some non-zero item. Spawn it.
  if (items_.at(currentIndex_).quantity > 255) {
    // We can only spawn up to 255 at a time.
    throw std::runtime_error("Cannot spawn enough items");
  }
  const PacketContainer packet = packet::building::ClientAgentOperatorRequest::makeItem(items_.at(currentIndex_).refItemId, items_.at(currentIndex_).quantity);
  VLOG(1) << "Sending packet for GM command to spawn " << items_.at(currentIndex_).quantity << " x " << bot_.gameData().getItemName(items_.at(currentIndex_).refItemId);
  bot_.packetBroker().injectPacket(packet, PacketContainer::Direction::kClientToServer);
  waitingForItemToSpawn_ = true;
  return Status::kNotDone;
}

} // namespace state::machine