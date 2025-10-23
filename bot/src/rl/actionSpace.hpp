#ifndef RL_ACTION_SPACE_HPP_
#define RL_ACTION_SPACE_HPP_

#include "rl/items.hpp"
#include "rl/skills.hpp"

#include <silkroad_lib/pk2/gameData.hpp>

#include <array>
#include <cstddef>

// Forward declarations
namespace rl {
class Action;
} // namespace rl
namespace state::machine {
class StateMachine;
} // namespace state::machine

namespace rl {

class ActionSpace {
public:
  // Returns the size of the action space.
  static constexpr size_t size() {
    return 1 /*sleep*/ + // IF-CHANGE: Sleep is supposed to be the first action in the action space.
           1 /*common attack*/ + // IF-CHANGE: CommonAttack is supposed to be the second action in the action space.
           kSkillIdsForObservations.size() +
           kItemIdsForObservations.size();
  }
  static std::unique_ptr<Action> buildAction(state::machine::StateMachine *parentStateMachine, const sro::pk2::GameData &gameData, sro::scalar_types::EntityGlobalId opponentGlobalId, size_t actionIndex);
private:
};

} // namespace rl

#endif // RL_ACTION_SPACE_HPP_