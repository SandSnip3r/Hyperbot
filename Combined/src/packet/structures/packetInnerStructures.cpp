#include "packetInnerStructures.hpp"

namespace packet::structures {

std::ostream& operator<<(std::ostream &stream, const ActionCommand &command) {
  stream << "CommandType: " << command.commandType;
  if (command.commandType == enums::CommandType::kExecute) {
    stream << ", ActionType: " << command.actionType;
    if (command.actionType == enums::ActionType::kCast || command.actionType == enums::ActionType::kDispel) {
      stream << ", refSkillId: " << command.refSkillId;
    }
    stream << ", targetType: " << command.targetType;
    if (command.targetType == enums::TargetType::kEntity) {
      stream << ", targetGlobalId: " << command.targetGlobalId;
    } else if (command.targetType == enums::TargetType::kLand) {
      stream << ", position: " << command.position;
    }
  }
  return stream;
}

} // namespace packet::structures