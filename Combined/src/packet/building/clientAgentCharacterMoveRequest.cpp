#include "clientAgentCharacterMoveRequest.hpp"
#include "../enums/packetEnums.hpp"

namespace packet::building {

enum class HasDestination : uint8_t {
  kNo = 0,
  kYes = 1
};

PacketContainer ClientAgentCharacterMoveRequest::moveTowardAngle(uint16_t angle) {
  StreamUtility stream;
  stream.Write<uint8_t>(static_cast<uint8_t>(HasDestination::kNo));
  stream.Write<uint8_t>(static_cast<uint8_t>(packet::enums::AngleAction::kGoForward));
  stream.Write<uint16_t>(angle);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

PacketContainer ClientAgentCharacterMoveRequest::moveToPosition(uint16_t regionId, uint32_t xOffset, uint32_t yOffset, uint32_t zOffset) {
  StreamUtility stream;
  stream.Write<uint8_t>(static_cast<uint8_t>(HasDestination::kYes));
  stream.Write<uint16_t>(regionId);
  if (regionId & 0x8000) {
    // Dungeon
    stream.Write<uint32_t>(xOffset);
    stream.Write<uint32_t>(yOffset);
    stream.Write<uint32_t>(zOffset);
  } else {
    // World
    stream.Write<uint16_t>(xOffset);
    stream.Write<uint16_t>(yOffset);
    stream.Write<uint16_t>(zOffset);
  }
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

} // namespace packet::building