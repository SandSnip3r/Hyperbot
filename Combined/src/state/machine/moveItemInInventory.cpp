#include "moveItemInInventory.hpp"

#include "bot.hpp"
#include "event/event.hpp"
#include "packet/building/clientAgentInventoryOperationRequest.hpp"

#include <absl/log/log.h>

#include <stdexcept>

namespace state::machine {

MoveItemInInventory::MoveItemInInventory(Bot &bot, uint8_t srcSlot, uint8_t destSlot) : StateMachine(bot), srcSlot_(srcSlot), destSlot_(destSlot) {
  stateMachineCreated(kName);
  // Prevent the human from moving anything
  pushBlockedOpcode(packet::Opcode::kClientAgentInventoryOperationRequest);
}

MoveItemInInventory::~MoveItemInInventory() {
  stateMachineDestroyed();
}

void MoveItemInInventory::onUpdate(const event::Event *event) {
  if (event != nullptr) {
    if (const auto *inventoryUpdatedEvent = reinterpret_cast<const event::InventoryUpdated*>(event)) {
      if (inventoryUpdatedEvent->srcSlotNum && *inventoryUpdatedEvent->srcSlotNum == srcSlot_) {
        // The target item moved
        waitingForItemToMove_ = false;
        if (!inventoryUpdatedEvent->destSlotNum) {
          throw std::runtime_error("Item was... dropped?");
        }
        if (*inventoryUpdatedEvent->destSlotNum == destSlot_) {
          // Item was successfully moved
          done_ = true;
          return;
        } else {
          // Item was moved, update where it is and try again
          LOG(INFO) << "Item was moved to somewhere else";
          // Even though we prevent the human from moving items, in theory, this could trigger. Maybe an inventory operation request was sent before we were constructed
          srcSlot_ = *inventoryUpdatedEvent->destSlotNum;
        }
      }
    }
  }

  if (waitingForItemToMove_) {
    // Item hasnt moved yet, dont try again
    return;
  }

  const auto moveItemPacket = packet::building::ClientAgentInventoryOperationRequest::withinInventoryPacket(srcSlot_, destSlot_, 1); // TODO: Figure out quantity, for now, we assume it's an equipment
  bot_.packetBroker().injectPacket(moveItemPacket, PacketContainer::Direction::kClientToServer);
  waitingForItemToMove_ = true;
}

bool MoveItemInInventory::done() const {
  return done_;
}

} // namespace state::machine