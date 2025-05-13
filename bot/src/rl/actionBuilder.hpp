#ifndef RL_ACTION_BUILDER_HPP_
#define RL_ACTION_BUILDER_HPP_

#include "rl/action.hpp"

#include <silkroad_lib/scalar_types.hpp>

#include <memory>

namespace state::machine {
class StateMachine;
} // namespace state::machine

namespace rl {

class ActionBuilder {
public:
  static std::unique_ptr<Action> buildAction(state::machine::StateMachine *parentStateMachine, sro::scalar_types::EntityGlobalId opponentGlobalId, int actionIndex);
  static constexpr int actionSpaceSize() { return 35; } // TODO: If changed, also change rl::JaxInterface
};

} // namespace rl

#endif // RL_ACTION_BUILDER_HPP_