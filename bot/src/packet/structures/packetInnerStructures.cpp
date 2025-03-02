#include "packet/structures/packetInnerStructures.hpp"

#include <sstream>

namespace packet::structures {

std::string ActionCommand::toString() const {
  std::stringstream ss;
  ss << "commandType: " << commandType;
  if (commandType == enums::CommandType::kExecute) {
    ss << ", actionType: " << actionType;
    if (actionType == enums::ActionType::kCast || actionType == enums::ActionType::kDispel) {
      ss << ", refSkillId: " << refSkillId;
    }
    ss << ", TargetType: " << targetType;
    if (targetType == enums::TargetType::kEntity) {
      ss << ", targetGlobalId: " << targetGlobalId;
    } else if (targetType == enums::TargetType::kLand) {
      ss << ", position: " << position;
    }
  }
  ss << '}';
  return ss.str();
}

} // namespace packet::structures