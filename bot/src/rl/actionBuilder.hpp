#ifndef RL_ACTION_BUILDER_HPP_
#define RL_ACTION_BUILDER_HPP_

#include "rl/action.hpp"

#include <silkroad_lib/scalar_types.hpp>

#include <memory>

namespace event {
struct Event;
} // namespace event

class Bot;


namespace rl {

class ActionBuilder {
public:
  static std::unique_ptr<Action> buildAction(Bot &bot, const event::Event *event, sro::scalar_types::EntityGlobalId opponentGlobalId, int actionIndex);
  static constexpr int actionSpaceSize() { return 38; }
};

} // namespace rl

#endif // RL_ACTION_BUILDER_HPP_