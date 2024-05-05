#include "clientAgentAlchemyStoneRequest.hpp"

#include "packet/enums/packetEnums.hpp"

namespace packet::building {

// TODO: This file is copied and pasted from ClientAgentAlchemyElixirRequest. They're the same, apart from the opcode.

PacketContainer ClientAgentAlchemyStoneRequest::fuseStone(sro::scalar_types::StorageIndexType targetIndex, sro::scalar_types::StorageIndexType stoneIndex) {
  StreamUtility stream;
  stream.Write(packet::enums::AlchemyAction::kFuse);
  stream.Write(packet::enums::AlchemyType::kMagicStone);
  stream.Write<uint8_t>(2); // Number of relevant items (target item and stone)
  stream.Write(targetIndex);
  stream.Write(stoneIndex);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}


} // namespace packet::building