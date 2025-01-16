#include "clientAgentCosCommandRequest.hpp"

#include "packet/enums/packetEnums.hpp"

namespace packet::building {

PacketContainer ClientAgentCosCommandRequest::pickup(sro::scalar_types::EntityGlobalId cosGlobalId, sro::scalar_types::EntityGlobalId targetGlobalId) {
  StreamUtility stream;
  stream.Write(cosGlobalId);
  stream.Write(enums::CosCommandType::kPick);
  stream.Write(targetGlobalId);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

} // namespace packet::building