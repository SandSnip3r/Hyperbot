#ifndef RL_OBSERVATION_HPP_
#define RL_OBSERVATION_HPP_

#include <silkroad_lib/scalar_types.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace event {
struct Event;
} // namespace event

class Bot;

namespace rl {

class Observation {
public:
  Observation() = default;
  Observation(const Bot &bot, const event::Event *event, sro::scalar_types::EntityGlobalId opponentGlobalId);
  std::string toString() const;
// private:
  uint32_t ourCurrentHp_;
  uint32_t ourMaxHp_;
  uint32_t ourCurrentMp_;
  uint32_t ourMaxMp_;
  uint32_t opponentCurrentHp_;
  uint32_t opponentMaxHp_;
  uint32_t opponentCurrentMp_;
  uint32_t opponentMaxMp_;
  int hpPotionCount_;
  std::vector<int> skillCooldowns_;
  std::vector<int> itemCooldowns_;
};

} // namespace rl

#endif // RL_OBSERVATION_HPP_