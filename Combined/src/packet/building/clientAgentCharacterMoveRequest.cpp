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

PacketContainer ClientAgentCharacterMoveRequest::moveToPosition(const sro::Position &newPosition) {
  StreamUtility stream;
  stream.Write<uint8_t>(static_cast<uint8_t>(HasDestination::kYes));
  stream.Write<uint16_t>(newPosition.regionId());
  if (newPosition.isDungeon()) {
    // Dungeon
    // x,y,z are truncated, not rounded
    stream.Write<uint32_t>(static_cast<uint32_t>(newPosition.xOffset()));
    stream.Write<uint32_t>(static_cast<uint32_t>(newPosition.yOffset()));
    stream.Write<uint32_t>(static_cast<uint32_t>(newPosition.zOffset()));
  } else {
    // World
    // x,y,z are truncated, not rounded
    stream.Write<uint16_t>(static_cast<uint16_t>(newPosition.xOffset()));
    stream.Write<uint16_t>(static_cast<uint16_t>(newPosition.yOffset()));
    stream.Write<uint16_t>(static_cast<uint16_t>(newPosition.zOffset()));
  }
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

} // namespace packet::building