#include "bot.hpp"
#include "rl/observation.hpp"

#include <absl/strings/str_format.h>

namespace rl {

Observation::Observation(const Bot &bot, const event::Event *event, sro::scalar_types::EntityGlobalId opponentGlobalId) {
  if (!bot.selfState()) {
    throw std::runtime_error("Cannot get an observation without a self state");
  }
  ourHp_ = bot.selfState()->currentHp();
  ourMp_ = bot.selfState()->currentMp();

  std::shared_ptr<entity::Self> opponent = bot.entityTracker().getEntity<entity::Self>(opponentGlobalId);
  opponentHp_ = opponent->currentHp();
  opponentMp_ = opponent->currentMp();
}

std::string Observation::toString() const {
  return absl::StrFormat("{hp:%d, mp:%d, opponentHp:%d, opponentMp:%d}", ourHp_, ourMp_, opponentHp_, opponentMp_);
}

}