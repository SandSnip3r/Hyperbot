#include "serverAgentEntityUpdateState.hpp"
#include "packet/enums/packetEnums.hpp"

namespace packet::building {

PacketContainer ServerAgentEntityUpdateState::updateLifeState(sro::scalar_types::EntityGlobalId globalId, sro::entity::LifeState state) {
  StreamUtility stream;
  stream.Write<>(globalId);
  stream.Write<>(enums::StateType::kLifeState);
  stream.Write<>(static_cast<uint8_t>(state));
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

} // namespace packet::building