#ifndef RL_AI_BASE_INTELLIGENCE_HPP_
#define RL_AI_BASE_INTELLIGENCE_HPP_

#include "rl/action.hpp"

#include <silkroad_lib/scalar_types.hpp>

#include <memory>

namespace event {
struct Event;
} // namespace event

class Bot;

namespace rl::ai {

class BaseIntelligence {
public:
virtual std::unique_ptr<Action> selectAction(Bot &bot, const event::Event *event, sro::scalar_types::EntityGlobalId opponentGlobalId) = 0;
};

} // namespace rl::ai

#endif // RL_AI_BASE_INTELLIGENCE_HPP_