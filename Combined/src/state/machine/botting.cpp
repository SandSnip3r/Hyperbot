#include "botting.hpp"
#include "townlooping.hpp"
#include "training.hpp"
#include "walking.hpp"

#include "logging.hpp"

namespace state::machine {

Botting::Botting(Bot &bot) : StateMachine(bot) {
  stateMachineCreated(kName);
  trainingSpotCenter_ = sro::Position{24742, 1114.48f, 47.0569f, 898.176f }; // TODO: Need to get from botting config

  // TODO: Figure out what we should initialize our childState_ as
  //  For development, we assume we want to be training, so first walk to the training spot
  childState_ = std::make_unique<Walking>(bot_, trainingSpotCenter_);
  // childState_ = std::make_unique<Townlooping>(bot_);
}

Botting::~Botting() {
  stateMachineDestroyed();
}

void Botting::onUpdate(const event::Event *event) {
  if (!childState_) {
    throw std::runtime_error("Botting must always have a child state when onUpdate is called");
  }

  childState_->onUpdate(event);
  if (childState_->done()) {
    // Move on to the next thing
    if (dynamic_cast<Townlooping*>(childState_.get())) {
      // After the townloop, we will walk to the training area
      childState_.reset();
      childState_ = std::make_unique<Walking>(bot_, trainingSpotCenter_);
    } else if (dynamic_cast<Walking*>(childState_.get())) {
      // We either just walked to town or walked to our training spot
      //  For now, we assume that we walked to our training spot
      childState_.reset();
      childState_ = std::make_unique<Training>(bot_, trainingSpotCenter_);
    } else if (dynamic_cast<Training*>(childState_.get())) {
      // Done training, need to go back to town
      static const sro::Position kCenterOfJangan{25000, 951.0f, -33.0f, 1372.0f}; // TODO: Instead, navigate to first NPC of townloop
      childState_.reset();
      childState_ = std::make_unique<Walking>(bot_, kCenterOfJangan);
    } else {
      throw std::runtime_error("Botting's child state is not valid");
    }
    // Recurse so that we start the next state
    // TODO: Might not be ideal
    onUpdate(event);
  }
}

bool Botting::done() const {
  // Botting is never done
  return false;
}

} // namespace state::machine