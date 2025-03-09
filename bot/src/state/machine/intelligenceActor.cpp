#include "intelligenceActor.hpp"

#include "bot.hpp"
#include "packet/building/clientAgentInventoryItemUseRequest.hpp"
#include "packet/building/clientAgentActionCommandRequest.hpp"
#include "rl/action.hpp"
#include "type_id/categories.hpp"

#include <absl/log/log.h>

namespace state::machine {

IntelligenceActor::IntelligenceActor(StateMachine *parent, rl::ai::BaseIntelligence *intelligence, common::PvpDescriptor::PvpId pvpId, sro::scalar_types::EntityGlobalId opponentGlobalId) : StateMachine(parent), intelligence_(intelligence), pvpId_(pvpId), opponentGlobalId_(opponentGlobalId) {
  LOG(INFO) << "Instantiated " << intelligence->name() << " intelligence actor!";
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
  setChildStateMachine(intelligence_->selectAction(bot_, this, event, pvpId_, opponentGlobalId_));

  // Run one update on the child state machine to let it start.
  const Status status = childState_->onUpdate(event);
  if (status == Status::kDone) {
    // If the action immediately completes, deconstruct it.
    childState_.reset();
  }

  // We are never done.
  return Status::kNotDone;
}

} // namespace state::machine
