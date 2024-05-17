#include "botting.hpp"
#include "townlooping.hpp"
#include "training.hpp"
#include "walking.hpp"

#include "bot.hpp"
#include "type_id/categories.hpp"
#include "proto_convert/convert.hpp"

#include <absl/log/log.h>

namespace state::machine {

Botting::Botting(Bot &bot) : StateMachine(bot) {
  stateMachineCreated(kName);
  setTrainingSpotFromConfig();
  initializeChildState();
}

void Botting::setTrainingSpotFromConfig() {
  const auto *characterConfig = bot_.config().getCharacterConfig(bot_.worldState().selfState().name);
  if (characterConfig == nullptr) {
    throw std::runtime_error("Constructing a Botting state machine but have no character config");
  }
  trainingSpotCenter_ = proto_convert::protoToPosition(characterConfig->training_config().center());
  trainingRadius_ = characterConfig->training_config().radius();
  LOG(INFO) << absl::StreamFormat("Parsed training spot center from config as (%d,%f,%f,%f), and radius as %f", trainingSpotCenter_.regionId(), trainingSpotCenter_.xOffset(), trainingSpotCenter_.yOffset(), trainingSpotCenter_.zOffset(), trainingRadius_);
  trainingAreaGeometry_ = std::make_unique<entity::Circle>(trainingSpotCenter_, trainingRadius_);
}

void Botting::initializeChildState() {
  // if we're out of supplies, we must go to town
  //  there must already be some logic somewhere to determine if we need to go to town, reuse that
  // if we're already in town, initialize as Townlooping
  //  there's a chance we have everything we need, then Townlooping will immediately finish and we'll move onto the next state as per the normal flow
  if (bot_.selfState().inTown() || bot_.needToGoToTown()) {
    LOG(INFO) << "Initializing state as Townlooping";
    setChildStateMachine<Townlooping>();
  } else {
    LOG(INFO) << "Initializing state as Training";
    setChildStateMachine<Training>(trainingAreaGeometry_->clone());
  }
}

Botting::~Botting() {
  stateMachineDestroyed();
}

void Botting::onUpdate(const event::Event *event) {
  if (!childState_) {
    throw std::runtime_error("Botting must always have a child state when onUpdate is called");
  }

  // TODO: Handle config changes.

  childState_->onUpdate(event);
  if (childState_->done()) {
    // Move on to the next thing
    if (dynamic_cast<Townlooping*>(childState_.get())) {
      // Done with the townloop, start training
      static int townloopCount = 0;
      LOG(INFO) << "Townloop count: " << ++townloopCount;
      setChildStateMachine<Training>(trainingAreaGeometry_->clone());
    } else if (dynamic_cast<Training*>(childState_.get())) {
      // Done training, go back to town
      if (bot_.selfState().lifeState == sro::entity::LifeState::kDead) {
        // We died. For now, we let Townlooping figure out what to do, since Training didn't want to handle it.
      }
      setChildStateMachine<Townlooping>();
    } else {
      throw std::runtime_error("Botting's child state is not valid");
    }
    // We switched to a new child state; recurse so that we call onUpdate on the new state machine
    // TODO: Might not be ideal
    onUpdate(event);
  }
}

bool Botting::done() const {
  // Botting is never done
  return false;
}

} // namespace state::machine