#include "serverAgentInventoryOperationResponse.hpp"

#include "packet/building/commonBuilding.hpp"
#include "packet/enums/packetEnums.hpp"

namespace packet::building {

PacketContainer ServerAgentInventoryOperationResponse::addItemByServerPacket(uint8_t targetSlot, const storage::Item &item) {
  StreamUtility stream;
  stream.Write<uint8_t>(1); // Result: success
  stream.Write<>(static_cast<std::underlying_type_t<packet::enums::ItemMovementType>>(packet::enums::ItemMovementType::kAddItemByServer));
  stream.Write<>(targetSlot);
  stream.Write<uint8_t>(0); // reason? TODO: Check what this value is usually
  writeGenericItem(stream, item);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

} // namespace packet::building