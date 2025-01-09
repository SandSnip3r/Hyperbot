#include "clientAgentOperatorRequest.hpp"
#include "../enums/packetEnums.hpp"

namespace packet::building {

PacketContainer ClientAgentOperatorRequest::toggleInvisible() {
  StreamUtility stream;
  stream.Write(enums::OperatorCommand::kInvisible);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

PacketContainer ClientAgentOperatorRequest::makeItem(sro::scalar_types::ReferenceObjectId refItemId, uint8_t optLevelOrAmount) {
  StreamUtility stream;
  stream.Write(enums::OperatorCommand::kMakeItem);
  stream.Write(refItemId);
  stream.Write(optLevelOrAmount);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

} // namespace packet::building