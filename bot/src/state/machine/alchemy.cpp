#include "alchemy.hpp"

#include "bot.hpp"
#include "event/event.hpp"
#include "type_id/categories.hpp"
#include "packet/building/clientAgentAlchemyElixirRequest.hpp"
#include "packet/building/clientAgentAlchemyStoneRequest.hpp"
#include "packet/building/clientAgentOperatorRequest.hpp"
#include "packet/building/commonBuilding.hpp"
#include "state/machine/dropItem.hpp"
#include "state/machine/pickItem.hpp"
#include "state/machine/walking.hpp"

#include <silkroad_lib/position_math.hpp>

#include <absl/log/log.h>

namespace state::machine {

Alchemy::Alchemy(Bot &bot) : StateMachine(bot) {
  stateMachineCreated(kName);
  // Prevent the human from moving anything
  pushBlockedOpcode(packet::Opcode::kClientAgentInventoryOperationRequest);

  initialize();
}

Alchemy::~Alchemy() {
  stateMachineDestroyed();
}

void Alchemy::initialize() {
  startPosition_ = bot_.selfState()->position();
}

void Alchemy::onUpdate(const event::Event *event) {
  static constexpr const sro::scalar_types::ReferenceObjectId kElixirRefObjId = 3679;
  static constexpr const sro::scalar_types::ReferenceObjectId kPowderRefObjId = 3692;
  static constexpr const sro::scalar_types::ReferenceObjectId kLuckyStoneRefObjId = 6880;
  static constexpr const sro::scalar_types::ReferenceObjectId kBladeRefObjId = 134;

  if (event != nullptr) {
    if (event->eventCode == event::EventCode::kAlchemyCompleted) {
      if (alchemyTimedOutEventId_) {
        bot_.eventBroker().cancelDelayedEvent(*alchemyTimedOutEventId_);
        alchemyTimedOutEventId_.reset();
      }
    } else if (event->eventCode == event::EventCode::kAlchemyTimedOut) {
      LOG(INFO) << "Received alchemy timed out event";
      if (alchemyTimedOutEventId_) {
        alchemyTimedOutEventId_.reset();
      } else {
        LOG(INFO) << "Whoa! Have no event!";
      }
    } else if (event->eventCode == event::EventCode::kGmCommandTimedOut) {
      LOG(INFO) << "Received gm command timed out event";
      if (makeItemTimedOutEventId_) {
        makeItemTimedOutEventId_.reset();
      } else {
        LOG(INFO) << "Whoa! Have no event!";
      }
      waitingForCreatedItem_.reset();
    } else if (auto *inventoryUpdatedEvent = dynamic_cast<const event::InventoryUpdated*>(event); inventoryUpdatedEvent != nullptr) {
      if (!inventoryUpdatedEvent->srcSlotNum) {
        // New item appeared in our inventory, is it the item we just picked?
        // TODO: I think this should move into PickItem, maybe optionally checked.
        const auto *item = bot_.selfState()->inventory.getItem(*inventoryUpdatedEvent->destSlotNum);
        if (waitingForCreatedItem_) {
          if (item->refItemId == *waitingForCreatedItem_) {
            // This is the item we created.
            waitingForCreatedItem_.reset();
          }
        }
      }
    }
  }

  if (childState_) {
    // Have a child state, it takes priority
    childState_->onUpdate(event);
    if (childState_->done()) {
      childState_.reset();
    } else {
      // Dont execute anything else in this function until the child state is done
      return;
    }
  }

  if (event != nullptr) {
    if (auto *entitySpawnedEvent = dynamic_cast<const event::EntitySpawned*>(event); entitySpawnedEvent != nullptr) {
      std::shared_ptr<entity::Entity> entity = bot_.worldState().getEntity(entitySpawnedEvent->globalId);
      if (waitingForCreatedItem_ && entity->refObjId == *waitingForCreatedItem_) {
        const auto distanceToItem = sro::position_math::calculateDistance2d(bot_.selfState()->position(), entity->position());
        if (distanceToItem < 25.0) {
          // Only look at items spawning very close to us.
          if (makeItemTimedOutEventId_) {
            bot_.eventBroker().cancelDelayedEvent(*makeItemTimedOutEventId_);
            makeItemTimedOutEventId_.reset();
          }
          // This is the item we created. Pick it up.
          setChildStateMachine<PickItem>(entitySpawnedEvent->globalId);
          onUpdate(event);
        }
      }
    }
  }

  if (waitingForCreatedItem_) {
    // Still waiting on an item we created.
    return;
  }

  if (alchemyTimedOutEventId_) {
    return;
  }

  const double distance = sro::position_math::calculateDistance2d(startPosition_, bot_.selfState()->position());
  if (distance > 25.0) {
    // Need to walk back to center.
    setChildStateMachine<Walking>(std::vector<packet::building::NetworkReadyPosition>({packet::building::NetworkReadyPosition(startPosition_)}));
    onUpdate(event);
    return;
  }

  const auto &inventory = bot_.selfState()->inventory;
  const auto slotsWithBlade = bot_.selfState()->inventory.findItemsWithRefId(kBladeRefObjId);
  if (slotsWithBlade.empty()) {
    // Don't have a blade to work with. Make one.
    LOG(INFO) << "spawning a blade that is +" << nextBladePlusToSpawn_;
    const auto makeItemPacket = packet::building::ClientAgentOperatorRequest::makeItem(kBladeRefObjId, nextBladePlusToSpawn_);
    ++nextBladePlusToSpawn_;
    if (nextBladePlusToSpawn_ == 9) {
      nextBladePlusToSpawn_ = 1;
    }
    bot_.packetBroker().injectPacket(makeItemPacket, PacketContainer::Direction::kClientToServer);
    makeItemTimedOutEventId_ = bot_.eventBroker().publishDelayedEvent(std::chrono::milliseconds(kMakeItemTimedOutMs), event::EventCode::kGmCommandTimedOut);
    waitingForCreatedItem_ = kBladeRefObjId;
    return;
  }
  const storage::Item *item = inventory.getItem(slotsWithBlade.front());
  // What + is the blade?
  const storage::ItemEquipment *blade = dynamic_cast<const storage::ItemEquipment*>(item);
  if (blade == nullptr) {
    // Whoa, expected item at this slot to be a blade, but it's not even a piece of equipment!
    // TODO: Somehow throw an error.
    LOG(INFO) << "Expected item at this slot (" << slotsWithBlade.front() << ") to be a blade";
    done_ = true;
    return;
  }
  if (blade->optLevel >= goalOptLevel_) {
    // Blade is already where we want it. Done.
    done_ = true;
    return;
  }
  if (blade->optLevel == 0) {
    // We just failed. Let's drop this sword.
    LOG(INFO) << "Want to drop blade";
    setChildStateMachine<DropItem>(slotsWithBlade.front());
    return;
  }

  if constexpr (kUseLuckStones) {
    // Does the blade have any luck on it?
    bool bladeHasLuck{false};
    for (const auto &blue : blade->magicParams) {
      const auto &magicOption = bot_.gameData().magicOptionData().getMagicOptionById(blue.type);
      // Rather than string comparisons, we could use the 2-4 character string from Param1, but string comparison is actually what the client uses.
      if (magicOption.mOptName128 == "MATTR_LUCK") {
        bladeHasLuck = true;
        break;
      }
    }

    if (!bladeHasLuck) {
      // Need to apply luck.
      // Do we have any luck stones?
      const auto luckyStoneSlots = bot_.selfState()->inventory.findItemsWithRefId(kLuckyStoneRefObjId);
      if (luckyStoneSlots.empty()) {
        // Don't have any lucky stones.
        LOG(INFO) << "Making a lucky stone";
        const auto makeItemPacket = packet::building::ClientAgentOperatorRequest::makeItem(kLuckyStoneRefObjId, 0);
        bot_.packetBroker().injectPacket(makeItemPacket, PacketContainer::Direction::kClientToServer);
        makeItemTimedOutEventId_ = bot_.eventBroker().publishDelayedEvent(std::chrono::milliseconds(kMakeItemTimedOutMs), event::EventCode::kGmCommandTimedOut);
        waitingForCreatedItem_ = kLuckyStoneRefObjId;
        return;
      }

      const auto useStonePacket = packet::building::ClientAgentAlchemyStoneRequest::fuseStone(slotsWithBlade.front(), luckyStoneSlots.front());
      bot_.packetBroker().injectPacket(useStonePacket, PacketContainer::Direction::kClientToServer);
      alchemyTimedOutEventId_ = bot_.eventBroker().publishDelayedEvent(std::chrono::milliseconds(kAlchemyTimedOutMs), event::EventCode::kAlchemyTimedOut);
      return;
    }
  }

  // Look through our inventory and see if we have a matching elixir.
  std::optional<sro::scalar_types::StorageIndexType> elixirSlot;
  std::optional<sro::scalar_types::StorageIndexType> powderSlot;
  for (int i=0; i<inventory.size(); ++i) {
    if (inventory.hasItem(i)) {
      const storage::Item *item = inventory.getItem(i);
      if (type_id::categories::kElixir.contains(item->typeId())) {
        // Elixir
        const auto targetTypes = item->itemInfo->elixirTargetItemTypeId3s();
        for (const auto targetTypeId3 : targetTypes) {
          if (type_id::categories::kEquipment.subCategory(targetTypeId3).contains(blade->typeId())) {
            elixirSlot = i;
            break;
          }
        }
      } else if (type_id::categories::kLuckyPowder.contains(item->typeId())) {
        // Lucky powder
        if (item->itemInfo->param1 == blade->degree()) {
          // param1 is degree
          // Matches our item's degree
          powderSlot = i;
        }
      }
      if (elixirSlot && powderSlot) {
        // Found everything we need.
        break;
      }
    }
  }
  if (!elixirSlot) {
    // Didn't find an elixir for our item.
    const auto makeItemPacket = packet::building::ClientAgentOperatorRequest::makeItem(kElixirRefObjId, 0);
    bot_.packetBroker().injectPacket(makeItemPacket, PacketContainer::Direction::kClientToServer);
    makeItemTimedOutEventId_ = bot_.eventBroker().publishDelayedEvent(std::chrono::milliseconds(kMakeItemTimedOutMs), event::EventCode::kGmCommandTimedOut);
    waitingForCreatedItem_ = kElixirRefObjId;
    return;
  }
  if (!powderSlot) {
    // Didn't find an powder for our item.
    const auto makeItemPacket = packet::building::ClientAgentOperatorRequest::makeItem(kPowderRefObjId, 50);
    bot_.packetBroker().injectPacket(makeItemPacket, PacketContainer::Direction::kClientToServer);
    makeItemTimedOutEventId_ = bot_.eventBroker().publishDelayedEvent(std::chrono::milliseconds(kMakeItemTimedOutMs), event::EventCode::kGmCommandTimedOut);
    waitingForCreatedItem_ = kPowderRefObjId;
    return;
  }

  const auto useElixirPacket = packet::building::ClientAgentAlchemyElixirRequest::fuseElixir(slotsWithBlade.front(), *elixirSlot, {*powderSlot});
  bot_.packetBroker().injectPacket(useElixirPacket, PacketContainer::Direction::kClientToServer);
  alchemyTimedOutEventId_ = bot_.eventBroker().publishDelayedEvent(std::chrono::milliseconds(kAlchemyTimedOutMs), event::EventCode::kAlchemyTimedOut);

  // Figure out what kind of elixir we want.
  // Protector: (3,1,) 1 2 3 9 10 11
  // Weapon: (3,1,) 6
  // Shield: (3,1,) 4
  // Accessory: (3,1,) 5 12
}

bool Alchemy::done() const {
  return done_;
}

} // namespace state::machine