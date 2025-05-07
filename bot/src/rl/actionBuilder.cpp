#include "rl/actionBuilder.hpp"

namespace rl {

std::unique_ptr<Action> ActionBuilder::buildAction(state::machine::StateMachine *parentStateMachine, const event::Event *event, sro::scalar_types::EntityGlobalId opponentGlobalId, int actionIndex) {
  switch (actionIndex) {
    case 0:
      return std::make_unique<Sleep>(parentStateMachine);
    case 1:
      return std::make_unique<CommonAttack>(parentStateMachine, opponentGlobalId);
    // case 2:
    //   return std::make_unique<CancelAction>(parentStateMachine);
    case 2:
      return std::make_unique<TargetlessSkill>(parentStateMachine, 28);
    case 3:
      return std::make_unique<TargetlessSkill>(parentStateMachine, 131);
    case 4:
      return std::make_unique<TargetlessSkill>(parentStateMachine, 554);
    case 5:
      return std::make_unique<TargetlessSkill>(parentStateMachine, 1253);
    case 6:
      return std::make_unique<TargetlessSkill>(parentStateMachine, 1256);
    case 7:
      return std::make_unique<TargetlessSkill>(parentStateMachine, 1271);
    case 8:
      return std::make_unique<TargetlessSkill>(parentStateMachine, 1272);
    case 9:
      return std::make_unique<TargetlessSkill>(parentStateMachine, 1281);
    case 10:
      return std::make_unique<TargetlessSkill>(parentStateMachine, 1335);
    case 11:
      return std::make_unique<TargetlessSkill>(parentStateMachine, 1377);
    case 12:
      return std::make_unique<TargetlessSkill>(parentStateMachine, 1380);
    case 13:
      return std::make_unique<TargetlessSkill>(parentStateMachine, 1398);
    case 14:
      return std::make_unique<TargetlessSkill>(parentStateMachine, 1399);
    case 15:
      return std::make_unique<TargetlessSkill>(parentStateMachine, 1410);
    case 16:
      return std::make_unique<TargetlessSkill>(parentStateMachine, 1421);
    case 17:
      return std::make_unique<TargetlessSkill>(parentStateMachine, 1441);
    // case 19:
    //   return std::make_unique<TargetlessSkill>(parentStateMachine, 8312); // Fire Combustion - Firefly
    // case 20:
    //   return std::make_unique<TargetlessSkill>(parentStateMachine, 21209); // Vision Fire Combustion
    case 18:
      return std::make_unique<TargetlessSkill>(parentStateMachine, 30577);
    case 19:
      return std::make_unique<TargetedSkill>(parentStateMachine, 37, opponentGlobalId);
    case 20:
      return std::make_unique<TargetedSkill>(parentStateMachine, 114, opponentGlobalId);
    case 21:
      return std::make_unique<TargetedSkill>(parentStateMachine, 298, opponentGlobalId);
    case 22:
      return std::make_unique<TargetedSkill>(parentStateMachine, 300, opponentGlobalId);
    case 23:
      return std::make_unique<TargetedSkill>(parentStateMachine, 322, opponentGlobalId);
    case 24:
      return std::make_unique<TargetedSkill>(parentStateMachine, 339, opponentGlobalId);
    case 25:
      return std::make_unique<TargetedSkill>(parentStateMachine, 371, opponentGlobalId);
    case 26:
      return std::make_unique<TargetedSkill>(parentStateMachine, 588, opponentGlobalId);
    case 27:
      return std::make_unique<TargetedSkill>(parentStateMachine, 610, opponentGlobalId);
    case 28:
      return std::make_unique<TargetedSkill>(parentStateMachine, 644, opponentGlobalId);
    case 29:
      return std::make_unique<TargetedSkill>(parentStateMachine, 1315, opponentGlobalId);
    case 30:
      return std::make_unique<TargetedSkill>(parentStateMachine, 1343, opponentGlobalId);
    case 31:
      return std::make_unique<TargetedSkill>(parentStateMachine, 1449, opponentGlobalId);
    case 32:
      return std::make_unique<UseItem>(parentStateMachine, 5);
    case 33:
      return std::make_unique<UseItem>(parentStateMachine, 12);
    case 34:
      return std::make_unique<UseItem>(parentStateMachine, 56);
    default:
      throw std::runtime_error("Invalid action choice");
  }
}

} // namespace rl
