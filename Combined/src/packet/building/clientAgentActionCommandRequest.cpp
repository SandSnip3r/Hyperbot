#include "clientAgentActionCommandRequest.hpp"
#include "packet/building/commonBuilding.hpp"
#include "packet/enums/packetEnums.hpp"

namespace packet::building {

PacketContainer ClientAgentActionCommandRequest::cancel() {
  StreamUtility stream;
  stream.Write(enums::CommandType::kCancel);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

PacketContainer ClientAgentActionCommandRequest::attack(uint32_t targetGId) {
  StreamUtility stream;
  stream.Write(enums::CommandType::kExecute);
  stream.Write(enums::ActionType::kAttack);
  stream.Write(enums::TargetType::kEntity);
  stream.Write(targetGId);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

PacketContainer ClientAgentActionCommandRequest::pickup(uint32_t targetGId) {
  StreamUtility stream;
  stream.Write(enums::CommandType::kExecute);
  stream.Write(enums::ActionType::kPickup);
  stream.Write(enums::TargetType::kEntity);
  stream.Write(targetGId);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

PacketContainer ClientAgentActionCommandRequest::trace(uint32_t targetGId) {
  StreamUtility stream;
  stream.Write(enums::CommandType::kExecute);
  stream.Write(enums::ActionType::kTrace);
  stream.Write(enums::TargetType::kEntity);
  stream.Write(targetGId);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

PacketContainer ClientAgentActionCommandRequest::cast(uint32_t refSkillId) {
  StreamUtility stream;
  stream.Write(enums::CommandType::kExecute);
  stream.Write(enums::ActionType::kCast);
  stream.Write(refSkillId);
  stream.Write(enums::TargetType::kNone);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

PacketContainer ClientAgentActionCommandRequest::cast(uint32_t refSkillId, uint32_t targetGId) {
  StreamUtility stream;
  stream.Write(enums::CommandType::kExecute);
  stream.Write(enums::ActionType::kCast);
  stream.Write(refSkillId);
  stream.Write(enums::TargetType::kEntity);
  stream.Write(targetGId);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

PacketContainer ClientAgentActionCommandRequest::dispel(uint32_t refSkillId, uint32_t targetGId) {
  StreamUtility stream;
  stream.Write(enums::CommandType::kExecute);
  stream.Write(enums::ActionType::kDispel);
  stream.Write(refSkillId);
  stream.Write(enums::TargetType::kEntity);
  stream.Write(targetGId);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

PacketContainer ClientAgentActionCommandRequest::command(const structures::ActionCommand& actionCommand) {
  StreamUtility stream;
  stream.Write(actionCommand.commandType);
  if (actionCommand.commandType == enums::CommandType::kExecute) {
    stream.Write(actionCommand.actionType);
    if (actionCommand.actionType == enums::ActionType::kCast || actionCommand.actionType == enums::ActionType::kDispel)  {
      stream.Write(actionCommand.refSkillId);
    }
    stream.Write(actionCommand.targetType);
    if (actionCommand.targetType == enums::TargetType::kEntity) {
      stream.Write(actionCommand.targetGlobalId);
    } else if (actionCommand.targetType == enums::TargetType::kLand) {
      writePosition(stream, actionCommand.position);
    }
  }
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

} // namespace packet::building