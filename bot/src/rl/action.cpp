#include "action.hpp"

#include "bot.hpp"
#include "packet/building/clientAgentActionCommandRequest.hpp"
#include "packet/building/clientAgentInventoryItemUseRequest.hpp"

#include <silkroad_lib/game_constants.hpp>

namespace rl {

using namespace state::machine;

Action::~Action() {}

Status Sleep::onUpdate(const event::Event *event) {
  // Any time we receive an event, we're done. This action is not a true sleep, but a "sleep at most X".
  // If we have a pending sleep event and this isn't it triggering, cancel ours.
  if (eventId_) {
    if (event != nullptr && event->eventCode != event::EventCode::kTimeout || event->eventId != *eventId_) {
      // Another event came before ours. Cancel ours.
      bot_.eventBroker().cancelDelayedEvent(*eventId_);
    }
    eventId_.reset();
    return Status::kDone;
  }
  // Have not yet sent our event.
  if (!eventId_) {
    eventId_ = bot_.eventBroker().publishDelayedEvent(event::EventCode::kTimeout, std::chrono::milliseconds(kSleepDurationMs));
  }
  return Status::kNotDone;
}

Status CommonAttack::onUpdate(const event::Event *event) {
  if (!sentPacket_) {
    CHAR_VLOG(1) << "Common attacking opponent";
    injectPacket(packet::building::ClientAgentActionCommandRequest::attack(targetGlobalId_), PacketContainer::Direction::kBotToServer);
    sentPacket_ = true;
  }
  return Status::kDone;
}

Status CancelAction::onUpdate(const event::Event *event) {
  if (!sentPacket_) {
    CHAR_VLOG(1) << "Cancelling";
    injectPacket(packet::building::ClientAgentActionCommandRequest::cancel(), PacketContainer::Direction::kBotToServer);
    sentPacket_ = true;
  }
  return Status::kDone;
}

Status TargetlessSkill::onUpdate(const event::Event *event) {
  if (!sentPacket_) {
    CHAR_VLOG(1) << "Casting " << bot_.gameData().getSkillName(skillRefId_) << "(" << skillRefId_ << ")";
    injectPacket(packet::building::ClientAgentActionCommandRequest::cast(skillRefId_), PacketContainer::Direction::kBotToServer);
    sentPacket_ = true;
  }
  return Status::kDone;
}

Status TargetedSkill::onUpdate(const event::Event *event) {
  if (!sentPacket_) {
    CHAR_VLOG(1) << "Casting " << bot_.gameData().getSkillName(skillRefId_) << "(" << skillRefId_ << ") on opponent";
    injectPacket(packet::building::ClientAgentActionCommandRequest::cast(skillRefId_, targetGlobalId_), PacketContainer::Direction::kBotToServer);
    sentPacket_ = true;
  }
  return Status::kDone;
}

Status UseItem::onUpdate(const event::Event *event) {
  if (!sentPacket_) {
    sentPacket_ = true; // TODO: "sentPacket" is not an accurate variable name.
    std::optional<sro::scalar_types::StorageIndexType> slot = bot_.selfState()->inventory.findFirstItemWithRefId(itemRefId_);
    const type_id::TypeId itemTypeId = type_id::getTypeId(bot_.gameData().itemData().getItemById(itemRefId_));
    if (!slot.has_value()) {
      // Item is not in inventory. In order to allow the model to see that it is not possible, we will send a packet to use the item, but at an empty slot. The server should send back an error.
      slot = bot_.selfState()->inventory.firstFreeSlot(sro::game_constants::kFirstInventorySlot);
      if (!slot.has_value()) {
        throw std::runtime_error("No free inventory slot");
      }
    }
    CHAR_VLOG(1) << "Using item " << bot_.gameData().getItemName(itemRefId_) << " at slot " << static_cast<int>(*slot);
    injectPacket(packet::building::ClientAgentInventoryItemUseRequest::packet(*slot, itemTypeId), PacketContainer::Direction::kBotToServer);
  }
  return Status::kDone;
}

} // namespace rl