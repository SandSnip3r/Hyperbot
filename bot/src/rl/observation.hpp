#ifndef RL_OBSERVATION_HPP_
#define RL_OBSERVATION_HPP_

#include <silkroad_lib/scalar_types.hpp>

#include <cstdint>
#include <string>

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
  uint32_t ourHp_;
  uint32_t ourMp_;
  uint32_t opponentHp_;
  uint32_t opponentMp_;
};

} // namespace rl

#endif // RL_OBSERVATION_HPP_