#include "action.hpp"

#include "bot.hpp"
#include "packet/building/clientAgentActionCommandRequest.hpp"
#include "packet/building/clientAgentInventoryItemUseRequest.hpp"

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
    // CHAR_VLOG(1) << "Sleeping for " << kSleepDurationMs << "ms";
    eventId_ = bot_.eventBroker().publishDelayedEvent(event::EventCode::kTimeout, std::chrono::milliseconds(kSleepDurationMs));
  }
  return Status::kNotDone;
}

Status CommonAttack::onUpdate(const event::Event *event) {
  if (!sentPacket_) {
    CHAR_VLOG(1) << "Common attacking opponent";
    bot_.packetBroker().injectPacket(packet::building::ClientAgentActionCommandRequest::attack(targetGlobalId_), PacketContainer::Direction::kClientToServer);
    sentPacket_ = true;
  }
  return Status::kDone;
}

Status CancelAction::onUpdate(const event::Event *event) {
  if (!sentPacket_) {
    CHAR_VLOG(1) << "Cancelling";
    bot_.packetBroker().injectPacket(packet::building::ClientAgentActionCommandRequest::cancel(), PacketContainer::Direction::kClientToServer);
    sentPacket_ = true;
  }
  return Status::kDone;
}

Status TargetlessSkill::onUpdate(const event::Event *event) {
  if (!sentPacket_) {
    CHAR_VLOG(1) << "Casting " << bot_.gameData().getSkillName(skillRefId_);
    bot_.packetBroker().injectPacket(packet::building::ClientAgentActionCommandRequest::cast(skillRefId_), PacketContainer::Direction::kClientToServer);
    sentPacket_ = true;
  }
  return Status::kDone;
}

Status TargetedSkill::onUpdate(const event::Event *event) {
  if (!sentPacket_) {
    CHAR_VLOG(1) << "Casting " << bot_.gameData().getSkillName(skillRefId_) << " on opponent";
    bot_.packetBroker().injectPacket(packet::building::ClientAgentActionCommandRequest::cast(skillRefId_, targetGlobalId_), PacketContainer::Direction::kClientToServer);
    sentPacket_ = true;
  }
  return Status::kDone;
}

Status UseItem::onUpdate(const event::Event *event) {
  if (!sentPacket_) {
    sentPacket_ = true; // TODO: "sentPacket" is not an accurate variable name.
    const std::optional<sro::scalar_types::StorageIndexType> slot = bot_.selfState()->inventory.findFirstItemWithRefId(itemRefId_);
    if (!slot.has_value()) {
      LOG(WARNING) << "UseItem Action: Item not in inventory.";
      return Status::kDone;
    }
    const auto itemTypeId = bot_.selfState()->inventory.getItem(*slot)->typeId();
    CHAR_VLOG(1) << "Using item " << bot_.gameData().getItemName(itemRefId_) << " at slot " << static_cast<int>(*slot);
    bot_.packetBroker().injectPacket(packet::building::ClientAgentInventoryItemUseRequest::packet(*slot, itemTypeId), PacketContainer::Direction::kClientToServer);
  }
  return Status::kDone;
}

} // namespace rl