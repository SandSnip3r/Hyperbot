#include "clientAgentAlchemyElixirRequest.hpp"

#include "packet/enums/packetEnums.hpp"

namespace packet::building {

PacketContainer ClientAgentAlchemyElixirRequest::fuseElixir(sro::scalar_types::StorageIndexType targetIndex, sro::scalar_types::StorageIndexType elixirIndex, std::vector<sro::scalar_types::StorageIndexType> enhancerIndices) {
  StreamUtility stream;
  stream.Write(packet::enums::AlchemyAction::kFuse);
  stream.Write(packet::enums::AlchemyType::kElixir);
  stream.Write<uint8_t>(2 + enhancerIndices.size());
  stream.Write(targetIndex);
  stream.Write(elixirIndex);
  for (const auto idx : enhancerIndices) {
    stream.Write(idx);
  }
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}


} // namespace packet::building