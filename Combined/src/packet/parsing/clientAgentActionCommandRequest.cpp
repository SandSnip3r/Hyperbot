#include "clientAgentActionCommandRequest.hpp"

namespace packet::parsing {

ClientAgentActionCommandRequest::ClientAgentActionCommandRequest(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  actionCommand_.commandType = static_cast<enums::CommandType>(stream.Read<uint8_t>());
  if (actionCommand_.commandType == enums::CommandType::kExecute) {
    actionCommand_.actionType = static_cast<enums::ActionType>(stream.Read<uint8_t>());
    if (actionCommand_.actionType == enums::ActionType::kCast || actionCommand_.actionType == enums::ActionType::kDispel) {
      actionCommand_.refSkillId = stream.Read<uint32_t>();
    }
    actionCommand_.targetType = static_cast<enums::TargetType>(stream.Read<uint8_t>());
    if (actionCommand_.targetType == enums::TargetType::kEntity) {
      actionCommand_.targetGlobalId = stream.Read<uint32_t>();
    } else if (actionCommand_.targetType == enums::TargetType::kLand) {
      actionCommand_.regionId = stream.Read<uint16_t>();
      uint32_t xAsInt = stream.Read<uint32_t>();
      actionCommand_.x = reinterpret_cast<float&>(xAsInt);
      uint32_t yAsInt = stream.Read<uint32_t>();
      actionCommand_.y = reinterpret_cast<float&>(yAsInt);
      uint32_t zAsInt = stream.Read<uint32_t>();
      actionCommand_.z = reinterpret_cast<float&>(zAsInt);
    }
  }
}

structures::ActionCommand ClientAgentActionCommandRequest::actionCommand() const {
  return actionCommand_;
}

enums::CommandType ClientAgentActionCommandRequest::commandType() const {
  return actionCommand_.commandType;
}

enums::ActionType ClientAgentActionCommandRequest::actionType() const {
  return actionCommand_.actionType;
}

uint32_t ClientAgentActionCommandRequest::refSkillId() const {
  return actionCommand_.refSkillId;
}

enums::TargetType ClientAgentActionCommandRequest::targetType() const {
  return actionCommand_.targetType;
}

uint32_t ClientAgentActionCommandRequest::targetGlobalId() const {
  return actionCommand_.targetGlobalId;
}

uint16_t ClientAgentActionCommandRequest::regionId() const {
  return actionCommand_.regionId;
}

float ClientAgentActionCommandRequest::x() const {
  return actionCommand_.x;
}

float ClientAgentActionCommandRequest::y() const {
  return actionCommand_.y;
}

float ClientAgentActionCommandRequest::z() const {
  return actionCommand_.z;
}

} // namespace packet::parsing