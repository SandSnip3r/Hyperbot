#include "clientAgentInventoryRepairRequest.hpp"

#include "packet/enums/packetEnums.hpp"

namespace packet::building {

PacketContainer ClientAgentInventoryRepairRequest::repairAllPacket(uint32_t gId) {
  StreamUtility stream;
  stream.Write<uint32_t>(gId);
  stream.Write<>(static_cast<std::underlying_type_t<packet::enums::RepairType>>(packet::enums::RepairType::kRepairAll));
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

} // namespace packet::building