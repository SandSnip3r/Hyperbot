#include "sellingItems.hpp"

#include "bot.hpp"
#include "event/event.hpp"
#include "packet/building/clientAgentInventoryOperationRequest.hpp"
#include "packet/building/serverAgentInventoryOperationResponse.hpp"

#include <absl/log/log.h>

namespace state::machine {

SellingItems::SellingItems(Bot &bot, const std::vector<sro::scalar_types::StorageIndexType> &slotsToSell) : StateMachine(bot), slotsToSell_(slotsToSell) {
  stateMachineCreated(kName);
  // We must be talking to an NPC at this point
  // Prevent the client from closing the talk dialog
  pushBlockedOpcode(packet::Opcode::kClientAgentActionDeselectRequest);
  // Prevent the client from moving items in inventory
  pushBlockedOpcode(packet::Opcode::kClientAgentInventoryOperationRequest);
  VLOG(2) << "Constructed SellingItems";
}

SellingItems::~SellingItems() {
  stateMachineDestroyed();
}

Status SellingItems::onUpdate(const event::Event *event) {
  VLOG(2) << "OnUpdate";
  if (slotsToSell_.empty()) {
    LOG(WARNING) << "No items to sell!";
    return Status::kDone;
  }

  if (event) {
    if (auto *inventoryUpdatedEvent = dynamic_cast<const event::InventoryUpdated*>(event)) {
      if (inventoryUpdatedEvent->srcSlotNum && inventoryUpdatedEvent->srcSlotNum == slotsToSell_[nextToSellIndex_] && !inventoryUpdatedEvent->destSlotNum) {
        // This seems to the item sell that we're expecting
        if (!waitingOnASell_) {
          throw std::runtime_error("Weird, the item at our target slot disappeared");
        }
        VLOG(1) << "Item successfully sold from slot " << static_cast<int>(slotsToSell_[nextToSellIndex_]);
        waitingOnASell_ = false;
        ++nextToSellIndex_;

        if (nextToSellIndex_ == slotsToSell_.size()) {
          VLOG(2) << "Done selling";
          return Status::kDone;
        }
      }
    }
  }

  uint16_t quantity{1};
  auto &inventory = bot_.selfState()->inventory;
  const auto currentSlot = slotsToSell_[nextToSellIndex_];
  if (!inventory.hasItem(currentSlot)) {
    throw std::runtime_error(absl::StrFormat("Trying to sell item at slot %d, but no item in this slot", currentSlot));
  }
  const auto *item = inventory.getItem(currentSlot);
  if (const auto *expItem = dynamic_cast<const storage::ItemExpendable*>(item)) {
    quantity = expItem->quantity;
  }
  VLOG(1) << absl::StreamFormat("Sell item at slot %d (x%d)", currentSlot, quantity);
  const auto sellPacket = packet::building::ClientAgentInventoryOperationRequest::sellPacket(currentSlot, quantity, bot_.selfState()->talkingGidAndOption->first);
  bot_.packetBroker().injectPacket(sellPacket, PacketContainer::Direction::kClientToServer);
  waitingOnASell_ = true;
  return Status::kNotDone;
}

} // namespace state::machine