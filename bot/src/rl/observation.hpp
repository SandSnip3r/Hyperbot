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
  int hpPotionCount_;
  std::array<int, 17> remainingTimeOurBuffs_;
  std::array<int, 17> remainingTimeOpponentBuffs_;
  std::array<int, 2> remainingTimeOurDebuffs_;
  std::array<int, 2> remainingTimeOpponentDebuffs_;
  std::array<int, 32> skillCooldowns_;
  std::array<int, 3> itemCooldowns_;
};

} // namespace rl

#endif // RL_OBSERVATION_HPP_