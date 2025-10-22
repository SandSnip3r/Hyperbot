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
  if (bot_.config() == nullptr) {
    throw std::runtime_error("Cannot construct Botting state machine if Bot does not have a config");
  }
  setTrainingSpotFromConfig();
  initializeChildState();
}

void Botting::setTrainingSpotFromConfig() {
  const proto::character_config::CharacterConfig &characterConfig = bot_.config()->proto();
  trainingSpotCenter_ = proto_convert::protoToPosition(characterConfig.training_config().center());
  trainingRadius_ = characterConfig.training_config().radius();
  LOG(INFO) << absl::StreamFormat("Parsed training spot center from config as (%d,%f,%f,%f), and radius as %f", trainingSpotCenter_.regionId(), trainingSpotCenter_.xOffset(), trainingSpotCenter_.yOffset(), trainingSpotCenter_.zOffset(), trainingRadius_);
  trainingAreaGeometry_ = std::make_unique<entity::Circle>(trainingSpotCenter_, trainingRadius_);
}

void Botting::initializeChildState() {
  std::shared_ptr<entity::Self> selfEntity = bot_.selfState();
  // if we're out of supplies, we must go to town
  //  there must already be some logic somewhere to determine if we need to go to town, reuse that
  // if we're already in town, initialize as Townlooping
  //  there's a chance we have everything we need, then Townlooping will immediately finish and we'll move onto the next state as per the normal flow
  if (selfEntity->inTown() || bot_.needToGoToTown()) {
    LOG(INFO) << "Initializing state as Townlooping";
    setChild<Townlooping>();
  } else {
    LOG(INFO) << "Initializing state as Training";
    setChild<Training>(trainingAreaGeometry_->clone());
  }
}

Botting::~Botting() {}

Status Botting::onUpdate(const event::Event *event) {
  if (!haveChild()) {
    throw std::runtime_error("Botting must always have a child state when onUpdate is called");
  }

  // TODO: Handle config changes.

  const Status status = onUpdateChild(event);
  if (status == Status::kDone) {
    // Move on to the next thing
    if (childIsType<Townlooping>()) {
      // Done with the townloop, start training
      static int townloopCount = 0;
      LOG(INFO) << "Townloop count: " << ++townloopCount;
      setChild<Training>(trainingAreaGeometry_->clone());
    } else if (childIsType<Training>()) {
      // Done training, go back to town
      std::shared_ptr<entity::Self> selfEntity = bot_.selfState();
      if (selfEntity->lifeState == sro::entity::LifeState::kDead) {
        // We died. For now, we let Townlooping figure out what to do, since Training didn't want to handle it.
      }
      setChild<Townlooping>();
    } else {
      throw std::runtime_error("Botting's child state is not valid");
    }
    // We switched to a new child state; recurse so that we call onUpdate on the new state machine
    // TODO: Might not be ideal
    return onUpdate(event);
  }

  return Status::kNotDone;
}

} // namespace state::machine