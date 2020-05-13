#include "clientAgentActionCommandRequest.hpp"
#include "../enums/packetEnums.hpp"

namespace packet::building {

PacketContainer ClientAgentActionCommandRequest::cancel() {
  StreamUtility stream;
  stream.Write<uint8_t>(static_cast<uint8_t>(enums::CommandType::kCancel));
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

PacketContainer ClientAgentActionCommandRequest::attack(uint32_t targetGId) {
  StreamUtility stream;
  stream.Write<uint8_t>(static_cast<uint8_t>(enums::CommandType::kExecute));
  stream.Write<uint8_t>(static_cast<uint8_t>(enums::ActionType::kAttack));
  stream.Write<uint8_t>(static_cast<uint8_t>(enums::TargetType::kEntity));
  stream.Write<uint32_t>(targetGId);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

PacketContainer ClientAgentActionCommandRequest::pickup(uint32_t targetGId) {
  StreamUtility stream;
  stream.Write<uint8_t>(static_cast<uint8_t>(enums::CommandType::kExecute));
  stream.Write<uint8_t>(static_cast<uint8_t>(enums::ActionType::kPickup));
  stream.Write<uint8_t>(static_cast<uint8_t>(enums::TargetType::kEntity));
  stream.Write<uint32_t>(targetGId);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

PacketContainer ClientAgentActionCommandRequest::trace(uint32_t targetGId) {
  StreamUtility stream;
  stream.Write<uint8_t>(static_cast<uint8_t>(enums::CommandType::kExecute));
  stream.Write<uint8_t>(static_cast<uint8_t>(enums::ActionType::kTrace));
  stream.Write<uint8_t>(static_cast<uint8_t>(enums::TargetType::kEntity));
  stream.Write<uint32_t>(targetGId);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

PacketContainer ClientAgentActionCommandRequest::cast(uint32_t refSkillId) {
  StreamUtility stream;
  stream.Write<uint8_t>(static_cast<uint8_t>(enums::CommandType::kExecute));
  stream.Write<uint8_t>(static_cast<uint8_t>(enums::ActionType::kCast));
  stream.Write<uint32_t>(refSkillId);
  stream.Write<uint8_t>(static_cast<uint8_t>(enums::TargetType::kNone));
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

PacketContainer ClientAgentActionCommandRequest::cast(uint32_t refSkillId, uint32_t targetGId) {
  StreamUtility stream;
  stream.Write<uint8_t>(static_cast<uint8_t>(enums::CommandType::kExecute));
  stream.Write<uint8_t>(static_cast<uint8_t>(enums::ActionType::kCast));
  stream.Write<uint32_t>(refSkillId);
  stream.Write<uint8_t>(static_cast<uint8_t>(enums::TargetType::kEntity));
  stream.Write<uint32_t>(targetGId);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

PacketContainer ClientAgentActionCommandRequest::dispel(uint32_t refSkillId, uint32_t targetGId) {
  StreamUtility stream;
  stream.Write<uint8_t>(static_cast<uint8_t>(enums::CommandType::kExecute));
  stream.Write<uint8_t>(static_cast<uint8_t>(enums::ActionType::kDispel));
  stream.Write<uint32_t>(refSkillId);
  stream.Write<uint8_t>(static_cast<uint8_t>(enums::TargetType::kEntity));
  stream.Write<uint32_t>(targetGId);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

} // namespace packet::building