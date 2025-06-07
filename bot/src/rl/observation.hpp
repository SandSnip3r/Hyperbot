#ifndef RL_OBSERVATION_HPP_
#define RL_OBSERVATION_HPP_

#include "event/eventCode.hpp"

#include <silkroad_lib/scalar_types.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <string>

class Bot;

namespace event {
class Event;
} // namespace event

namespace rl {

class Observation {
public:
  Observation() = default;
  Observation(const Bot &bot, const event::Event *event, sro::scalar_types::EntityGlobalId opponentGlobalId);
  std::string toString() const;
// private:
  std::chrono::high_resolution_clock::time_point timestamp_;
  event::EventCode eventCode_;
  uint32_t ourCurrentHp_;
  uint32_t ourMaxHp_;
  uint32_t ourCurrentMp_;
  uint32_t ourMaxMp_;
  bool weAreKnockedDown_;
  uint32_t opponentCurrentHp_;
  uint32_t opponentMaxHp_;
  uint32_t opponentCurrentMp_;
  uint32_t opponentMaxMp_;
  bool opponentIsKnockedDown_;
  int hpPotionCount_; // TODO: Normalize here rather than in JaxInterface

  // Normalized to [0,1] for buff duration.
  std::array<float, 17> remainingTimeOurBuffs_;
  std::array<float, 17> remainingTimeOpponentBuffs_;

  // Normalized to [0,1] for debuff duration. For now, this only refers to legacy debuffs.
  std::array<float, 12> remainingTimeOurDebuffs_;
  std::array<float, 12> remainingTimeOpponentDebuffs_;

  // Normalized to [0,1] for skill cooldown.
  std::array<float, 32> skillCooldowns_;

  // Normalized to [0,1] for item cooldown.
  std::array<float, 3> itemCooldowns_;
};

} // namespace rl

#endif // RL_OBSERVATION_HPP_