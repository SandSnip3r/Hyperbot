#include "intelligenceActor.hpp"

#include "bot.hpp"
#include "packet/building/clientAgentInventoryItemUseRequest.hpp"
#include "packet/building/clientAgentActionCommandRequest.hpp"
#include "rl/action.hpp"
#include "type_id/categories.hpp"

#include <absl/log/log.h>

namespace state::machine {

IntelligenceActor::IntelligenceActor(Bot &bot, rl::ai::BaseIntelligence *intelligence, sro::scalar_types::EntityGlobalId opponentGlobalId) : StateMachine(bot), intelligence_(intelligence), opponentGlobalId_(opponentGlobalId) {
  LOG(INFO) << "Instantiated intelligence actor!";
}

IntelligenceActor::~IntelligenceActor() {
}

Status IntelligenceActor::onUpdate(const event::Event *event) {
  if (childState_ != nullptr) {
    // The child state machine didn't immediately finish.
    // Run the update.
    const Status status = childState_->onUpdate(event);
    if (status == Status::kNotDone) {
      // Child state is not done, nothing to do for now.
      return status;
    }
    // Child state is done, reset it then continue to get our next action.
    childState_.reset();
  }

  // Since actions are state machines, immediately set the selected action as our current active child state machine.
  setChildStateMachine(intelligence_->selectAction(bot_, event, opponentGlobalId_));

  // Run one update on the child state machine to let it start.
  const Status status = childState_->onUpdate(event);
  if (status == Status::kDone) {
    // If the action immediately completes, deconstruct it.
    childState_.reset();
  }

  // We are never done.
  return Status::kNotDone;
}

void IntelligenceActor::useItem(sro::pk2::ref::ItemId refId) {
  // Find this item in our inventory.
  for (int slot=0; slot<bot_.inventory().size(); ++slot) {
    if (!bot_.inventory().hasItem(slot)) {
      // No item here.
      continue;
    }
    const storage::Item *item = bot_.inventory().getItem(slot);
    if (item->refItemId == refId) {
      // Use this item.
      CHAR_VLOG(1) << "Sending packet to use item at slot " << slot;
      bot_.packetBroker().injectPacket(packet::building::ClientAgentInventoryItemUseRequest::packet(slot, item->typeId()), PacketContainer::Direction::kBotToServer);
      break;
    }
  }
}

} // namespace state::machine
