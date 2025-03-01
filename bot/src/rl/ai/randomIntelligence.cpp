#include "rl/ai/randomIntelligence.hpp"

namespace rl::ai {

std::unique_ptr<Action> RandomIntelligence::selectAction(Bot &bot, const event::Event *event, sro::scalar_types::EntityGlobalId opponentGlobalId) {
  // Start with a high probability to do nothing/sleep.
  std::bernoulli_distribution sleepDist(0.85);
  if (sleepDist(randomEngine_)) {
    return std::make_unique<Sleep>(bot);
  }

  // Not sleeping, choose a true random action.
  std::uniform_int_distribution<int> actionDist(0, 37);
  const int actionChoice = actionDist(randomEngine_);
  switch (actionChoice) {
    case 0:
      return std::make_unique<Sleep>(bot);
    case 1:
      return std::make_unique<CommonAttack>(bot, opponentGlobalId);
    case 2:
      return std::make_unique<CancelAction>(bot);
    case 3:
      return std::make_unique<TargetlessSkill>(bot, 28);
    case 4:
      return std::make_unique<TargetlessSkill>(bot, 131);
    case 5:
      return std::make_unique<TargetlessSkill>(bot, 554);
    case 6:
      return std::make_unique<TargetlessSkill>(bot, 1253);
    case 7:
      return std::make_unique<TargetlessSkill>(bot, 1256);
    case 8:
      return std::make_unique<TargetlessSkill>(bot, 1271);
    case 9:
      return std::make_unique<TargetlessSkill>(bot, 1272);
    case 10:
      return std::make_unique<TargetlessSkill>(bot, 1281);
    case 11:
      return std::make_unique<TargetlessSkill>(bot, 1335);
    case 12:
      return std::make_unique<TargetlessSkill>(bot, 1377);
    case 13:
      return std::make_unique<TargetlessSkill>(bot, 1380);
    case 14:
      return std::make_unique<TargetlessSkill>(bot, 1398);
    case 15:
      return std::make_unique<TargetlessSkill>(bot, 1399);
    case 16:
      return std::make_unique<TargetlessSkill>(bot, 1410);
    case 17:
      return std::make_unique<TargetlessSkill>(bot, 1421);
    case 18:
      return std::make_unique<TargetlessSkill>(bot, 1441);
    case 19:
      return std::make_unique<TargetlessSkill>(bot, 8312);
    case 20:
      return std::make_unique<TargetlessSkill>(bot, 21209);
    case 21:
      return std::make_unique<TargetlessSkill>(bot, 30577);
    case 22:
      return std::make_unique<TargetedSkill>(bot, 37, opponentGlobalId);
    case 23:
      return std::make_unique<TargetedSkill>(bot, 114, opponentGlobalId);
    case 24:
      return std::make_unique<TargetedSkill>(bot, 298, opponentGlobalId);
    case 25:
      return std::make_unique<TargetedSkill>(bot, 300, opponentGlobalId);
    case 26:
      return std::make_unique<TargetedSkill>(bot, 322, opponentGlobalId);
    case 27:
      return std::make_unique<TargetedSkill>(bot, 339, opponentGlobalId);
    case 28:
      return std::make_unique<TargetedSkill>(bot, 371, opponentGlobalId);
    case 29:
      return std::make_unique<TargetedSkill>(bot, 588, opponentGlobalId);
    case 30:
      return std::make_unique<TargetedSkill>(bot, 610, opponentGlobalId);
    case 31:
      return std::make_unique<TargetedSkill>(bot, 644, opponentGlobalId);
    case 32:
      return std::make_unique<TargetedSkill>(bot, 1315, opponentGlobalId);
    case 33:
      return std::make_unique<TargetedSkill>(bot, 1343, opponentGlobalId);
    case 34:
      return std::make_unique<TargetedSkill>(bot, 1449, opponentGlobalId);
    case 35:
      return std::make_unique<UseItem>(bot, 5);
    case 36:
      return std::make_unique<UseItem>(bot, 12);
    case 37:
      return std::make_unique<UseItem>(bot, 56);
    default:
      throw std::runtime_error("Invalid action choice");
  }
}

} // namespace rl::ai