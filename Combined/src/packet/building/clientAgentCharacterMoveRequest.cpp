#include "clientAgentCharacterMoveRequest.hpp"
#include "../enums/packetEnums.hpp"

namespace packet::building {

NetworkReadyPosition::NetworkReadyPosition(const sro::Position &pos) : convertedPosition_(truncateForNetwork(pos)) {
}

sro::Position NetworkReadyPosition::asSroPosition() const {
  return convertedPosition_;
}

sro::Position NetworkReadyPosition::truncateForNetwork(const sro::Position &pos) {
  // x,y,z are truncated, not rounded
  if (pos.isDungeon()) {
    return {
              pos.regionId(),
              static_cast<float>(static_cast<uint32_t>(pos.xOffset())),
              static_cast<float>(static_cast<uint32_t>(pos.yOffset())),
              static_cast<float>(static_cast<uint32_t>(pos.zOffset()))
           };
  } else {
    return {
              pos.regionId(),
              static_cast<float>(static_cast<uint16_t>(pos.xOffset())),
              static_cast<float>(static_cast<uint16_t>(pos.yOffset())),
              static_cast<float>(static_cast<uint16_t>(pos.zOffset()))
           };
  }
}

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

PacketContainer ClientAgentCharacterMoveRequest::moveToPosition(const NetworkReadyPosition &newPosition) {
  StreamUtility stream;
  stream.Write<uint8_t>(static_cast<uint8_t>(HasDestination::kYes));
  const auto &newPositionAsSroPosition = newPosition.asSroPosition();
  stream.Write<uint16_t>(newPositionAsSroPosition.regionId());
  if (newPositionAsSroPosition.isDungeon()) {
    // Dungeon
    // x,y,z are truncated, not rounded
    stream.Write<uint32_t>(static_cast<uint32_t>(newPositionAsSroPosition.xOffset()));
    stream.Write<uint32_t>(static_cast<uint32_t>(newPositionAsSroPosition.yOffset()));
    stream.Write<uint32_t>(static_cast<uint32_t>(newPositionAsSroPosition.zOffset()));
  } else {
    // World
    // x,y,z are truncated, not rounded
    stream.Write<uint16_t>(static_cast<uint16_t>(newPositionAsSroPosition.xOffset()));
    stream.Write<uint16_t>(static_cast<uint16_t>(newPositionAsSroPosition.yOffset()));
    stream.Write<uint16_t>(static_cast<uint16_t>(newPositionAsSroPosition.zOffset()));
  }
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

} // namespace packet::building