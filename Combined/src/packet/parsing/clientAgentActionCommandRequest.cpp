#include "clientAgentActionCommandRequest.hpp"
#include "commonParsing.hpp"

namespace packet::parsing {

ClientAgentActionCommandRequest::ClientAgentActionCommandRequest(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  actionCommand_.commandType = static_cast<enums::CommandType>(stream.Read<uint8_t>());
  if (actionCommand_.commandType == enums::CommandType::kExecute) {
    actionCommand_.actionType = static_cast<enums::ActionType>(stream.Read<uint8_t>());
    if (actionCommand_.actionType == enums::ActionType::kCast || actionCommand_.actionType == enums::ActionType::kDispel) {
      actionCommand_.refSkillId = stream.Read<sro::scalar_types::ReferenceObjectId>();
    }
    actionCommand_.targetType = static_cast<enums::TargetType>(stream.Read<uint8_t>());
    if (actionCommand_.targetType == enums::TargetType::kEntity) {
      actionCommand_.targetGlobalId = stream.Read<sro::scalar_types::EntityGlobalId>();
    } else if (actionCommand_.targetType == enums::TargetType::kLand) {
      actionCommand_.position = parsePosition(stream);
    }
  }
}

const structures::ActionCommand& ClientAgentActionCommandRequest::actionCommand() const {
  return actionCommand_;
}

} // namespace packet::parsing