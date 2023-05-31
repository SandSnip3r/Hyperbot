#include "botting.hpp"
#include "townlooping.hpp"
#include "training.hpp"
#include "walking.hpp"

#include "bot.hpp"
#include "logging.hpp"
#include "type_id/categories.hpp"

namespace state::machine {

const sro::Position Botting::kCenterOfJangan_{25000, 951.0f, -33.0f, 1372.0f};

Botting::Botting(Bot &bot) : StateMachine(bot) {
  stateMachineCreated(kName);
  // TODO: Need to get training spot from botting config.
  // trainingSpotCenter_ = sro::Position{24742, 977.0f, 56.501f, 1127.0f }; // Southwest of Jangan, no obstacles
  // trainingSpotCenter_ = sro::Position{24744, 1406.0f, -43.0f, 203.0f }; // South of Jangan, obstacles
  trainingSpotCenter_ = sro::Position{24232, 1617.0f, 11.0f, 1369.0f }; // Farther south of Jangan, obstacles
  constexpr double kMonsterRange{1800.0};
  trainingAreaGeometry_ = std::make_unique<entity::Circle>(trainingSpotCenter_, kMonsterRange);

  initializeChildState();
}

void Botting::initializeChildState() {
  // if we're out of supplies, we must go to town
  //  there must already be some logic somewhere to determine if we need to go to town, reuse that
  // if we're already in town, initialize as Townlooping
  //  there's a chance we have everything we need, then Townlooping will immediately finish and we'll move onto the next state as per the normal flow
  if (bot_.selfState().inTown() || bot_.needToGoToTown()) {
    LOG() << "Initializing state as Townlooping" << std::endl;
    setChildStateMachine<Townlooping>();
  } else {
    LOG() << "Initializing state as Training" << std::endl;
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

  childState_->onUpdate(event);
  if (childState_->done()) {
    // Move on to the next thing
    if (dynamic_cast<Townlooping*>(childState_.get())) {
      // Done with the townloop, start training
      static int townloopCount = 0;
      LOG() << "Townloop count: " << ++townloopCount << std::endl;
      setChildStateMachine<Training>(trainingAreaGeometry_->clone());
    } else if (dynamic_cast<Training*>(childState_.get())) {
      // Done training, go back to town
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