#include "moveItem.hpp"

#include "bot.hpp"
#include "event/event.hpp"
#include "packet/building/clientAgentInventoryOperationRequest.hpp"

#include <silkroad_lib/game_constants.hpp>

#include <absl/log/log.h>
#include <absl/strings/str_format.h>
#include <absl/strings/str_join.h>

#include <stdexcept>

namespace state::machine {

MoveItem::MoveItem(StateMachine *parent, sro::storage::Position source, sro::storage::Position destination) : StateMachine(parent), source_(source), destination_(destination) {
}

MoveItem::~MoveItem() {}

Status MoveItem::onUpdate(const event::Event *event) {
  if (!initialized_) {
    CHAR_VLOG(1) << absl::StreamFormat("MoveItem: Moving item from %s-%d to %s-%d", toString(source_.storage), source_.slotNum, toString(destination_.storage), destination_.slotNum);
    // Prevent the human from moving anything
    pushBlockedOpcode(packet::Opcode::kClientAgentInventoryOperationRequest);
    initialized_ = true;
  }
  if (event != nullptr) {
    CHAR_VLOG(3) << absl::StreamFormat("MoveItem: Received event %s", event::toString(event->eventCode));
    if (const event::ItemMoved *itemMovedEvent = dynamic_cast<const event::ItemMoved*>(event); itemMovedEvent != nullptr) {
      if (itemMovedEvent->globalId == bot_.selfState()->globalId) {
        if (itemMovedEvent->source && *itemMovedEvent->source == source_) {
          // The target item moved
          CHAR_VLOG(1) << absl::StreamFormat("Item moved from %s-%d to %s-%d", toString(source_.storage), source_.slotNum, toString(destination_.storage), destination_.slotNum);
          if (timeoutEventId_) {
            bot_.eventBroker().cancelDelayedEvent(*timeoutEventId_);
            timeoutEventId_.reset();
          }
          if (!itemMovedEvent->destination) {
            throw std::runtime_error("Item is gone");
          }
          if (*itemMovedEvent->destination != destination_) {
            throw std::runtime_error("Item was moved to somewhere else");
          }
          // Item was moved to our specified destination position.
          return Status::kDone;
        } else if (itemMovedEvent->destination && *itemMovedEvent->destination == destination_) {
          sro::scalar_types::StorageIndexType destinationSlot = itemMovedEvent->destination->slotNum;
          throw std::runtime_error(absl::StrFormat("Some other item %s was moved to our destination slot %d", bot_.gameData().getItemName(bot_.inventory().getItem(destinationSlot)->refItemId), destinationSlot));
        } else {
          CHAR_VLOG(2) << absl::StreamFormat("MoveItem: Our item moved, but not from our source %s-%d to our destination %s-%d", toString(source_.storage), source_.slotNum, toString(destination_.storage), destination_.slotNum);
          CHAR_VLOG(2) << "  src? " << (itemMovedEvent->source.has_value() ? itemMovedEvent->source->toString() : std::string("NONE")) << ", dest? " << (itemMovedEvent->destination.has_value() ? itemMovedEvent->destination->toString() : std::string("NONE"));
        }
      } else {
        std::string charName = "<unknown>";
        std::shared_ptr<entity::Self> other = bot_.entityTracker().getEntity<entity::Self>(itemMovedEvent->globalId);
        if (other) {
          charName = other->name;
        }
        CHAR_VLOG(3) << absl::StreamFormat("MoveItem: Someone else's item moved %s (name: %s)", itemMovedEvent->globalId, charName);
      }
    } else if (const event::ItemMoveFailed *itemMoveFailedEvent = dynamic_cast<const event::ItemMoveFailed*>(event); itemMoveFailedEvent != nullptr) {
      if (itemMoveFailedEvent->globalId == bot_.selfState()->globalId) {
        // Reset and try again.
        if (timeoutEventId_) {
          bot_.eventBroker().cancelDelayedEvent(*timeoutEventId_);
          timeoutEventId_.reset();
        }
        if (attemptCount_ >= kMaxAttempts) {
          throw std::runtime_error(absl::StrFormat("[%s] Item move operation failed with error %d after %d retries", kName, itemMoveFailedEvent->errorCode, attemptCount_));
        }
        CHAR_VLOG(1) << absl::StreamFormat("Item move operation failed with error %d. Retrying... (attempt %d/%d)", itemMoveFailedEvent->errorCode, attemptCount_, kMaxAttempts);
      }
    } else if (event->eventCode == event::EventCode::kTimeout && timeoutEventId_ && *timeoutEventId_ == event->eventId) {
      timeoutEventId_.reset();
      if (attemptCount_ >= kMaxAttempts) {
        throw std::runtime_error(absl::StrFormat("[%s] Item move operation failed with timeout after %d retries", kName, attemptCount_));
      }
      CHAR_VLOG(1) << absl::StreamFormat("MoveItem timed out. Retrying... (attempt %d/%d)", attemptCount_, kMaxAttempts);
    }
  }

  if (timeoutEventId_) {
    // Still waiting on the item to move.
    return Status::kNotDone;
  }

  const storage::Item *item = bot_.inventory().getItem(source_.slotNum);
  if (item == nullptr) {
    throw std::runtime_error(absl::StrFormat("MoveItem::Item at slot %d is null for char %s", source_.slotNum, bot_.selfState()->name));
  }
  const storage::ItemEquipment *equipment = dynamic_cast<const storage::ItemEquipment*>(item);
  if (equipment == nullptr) {
    throw std::runtime_error("Item is not an equipment, for now, that is the only type which is handled");
  }

  if (source_.storage == sro::storage::Storage::kAvatarInventory && destination_.storage == sro::storage::Storage::kInventory) {
    // In SRO, the only possible destination is the first free slot in the inventory.
    std::optional<sro::scalar_types::StorageIndexType> firstFreeSlot = bot_.inventory().firstFreeSlot(sro::game_constants::kFirstInventorySlot);
    if (firstFreeSlot) {
      if (destination_.slotNum != *firstFreeSlot) {
        sro::scalar_types::StorageIndexType low = std::min(destination_.slotNum, *firstFreeSlot);
        sro::scalar_types::StorageIndexType high = std::max(destination_.slotNum, *firstFreeSlot);
        std::vector<const storage::Item*> items;
        for (sro::scalar_types::StorageIndexType slotNum = low; slotNum <= high; ++slotNum) {
          items.push_back(bot_.inventory().getItem(slotNum));
        }
        CHAR_LOG(WARNING) << absl::StreamFormat("Trying to unequip avatar item to inventory index %d, but, due to SRO, it will go to %d. Items in slots are [%s]", destination_.slotNum, *firstFreeSlot, absl::StrJoin(items, ", ", [this](std::string *out, const storage::Item *item) {
          if (item) {
            absl::StrAppend(out, bot_.gameData().getItemName(item->refItemId));
          } else {
            absl::StrAppend(out, "null");
          }
        }));
      }
    } else {
      throw std::runtime_error("No free slot available in inventory for removing avatar item");
    }
  }

  PacketContainer moveItemPacket;
  if (source_.storage == sro::storage::Storage::kInventory && destination_.storage == sro::storage::Storage::kInventory) {
    // Move item within inventory
    CHAR_VLOG(1) << absl::StreamFormat("Moving item within inventory from %d to %d", source_.slotNum, destination_.slotNum);
    moveItemPacket = packet::building::ClientAgentInventoryOperationRequest::withinInventoryPacket(source_.slotNum, destination_.slotNum, /*quantity=*/1);
  } else if (source_.storage == sro::storage::Storage::kInventory && destination_.storage == sro::storage::Storage::kAvatarInventory) {
    // Move item from inventory to avatar
    CHAR_VLOG(1) << absl::StreamFormat("Moving item from inventory-%d to avatar-%d", source_.slotNum, destination_.slotNum);
    moveItemPacket = packet::building::ClientAgentInventoryOperationRequest::inventoryToAvatarPacket(source_.slotNum, destination_.slotNum);
  } else if (source_.storage == sro::storage::Storage::kAvatarInventory && destination_.storage == sro::storage::Storage::kInventory) {
    // Move item from avatar to inventory
    CHAR_VLOG(1) << absl::StreamFormat("Moving item from avatar-%d to inventory-%d", source_.slotNum, destination_.slotNum);
    moveItemPacket = packet::building::ClientAgentInventoryOperationRequest::avatarToInventoryPacket(source_.slotNum, destination_.slotNum);
  } else {
    throw std::runtime_error(absl::StrFormat("Unhandled source %s and destination %s", toString(source_.storage), toString(destination_.storage)));
  }
  injectPacket(moveItemPacket, PacketContainer::Direction::kBotToServer);
  timeoutEventId_ = bot_.eventBroker().publishDelayedEvent(event::EventCode::kTimeout, std::chrono::milliseconds(1000));
  ++attemptCount_;
  return Status::kNotDone;
}

} // namespace state::machine