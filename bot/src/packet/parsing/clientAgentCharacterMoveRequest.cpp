#include "clientAgentCharacterMoveRequest.hpp"

#include <silkroad_lib/position_math.hpp>

#include <utility>

namespace packet::parsing {

ClientAgentCharacterMoveRequest::ClientAgentCharacterMoveRequest(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  bool hasDestination = stream.Read<uint8_t>();
  if (hasDestination) {
    sro::RegionId regionId;
    int32_t xOffset, yOffset, zOffset;
    regionId = stream.Read<sro::RegionId>();
    if (sro::position_math::regionIsDungeon(regionId)) {
      // Dungeon
      xOffset = stream.Read<int32_t>();
      yOffset = stream.Read<int32_t>();
      zOffset = stream.Read<int32_t>();
    } else {
      // World
      xOffset = stream.Read<int16_t>();
      yOffset = stream.Read<int16_t>();
      zOffset = stream.Read<int16_t>();
    }
    destinationPosition_.emplace(regionId, xOffset, yOffset, zOffset);
  } else {
    angleAction_ = static_cast<packet::enums::AngleAction>(stream.Read<uint8_t>());
    angle_ = stream.Read<uint16_t>();
  }
}

bool ClientAgentCharacterMoveRequest::hasDestination() const {
  return destinationPosition_.has_value();
}

const sro::Position& ClientAgentCharacterMoveRequest::destinationPosition() const {
  if (!destinationPosition_) {
    throw std::runtime_error("ClientAgentCharacterMoveRequest: destinationPosition does not exist");
  }
  return destinationPosition_.value();
}

packet::enums::AngleAction ClientAgentCharacterMoveRequest::angleAction() const {
  return angleAction_;
}

uint16_t ClientAgentCharacterMoveRequest::angle() const {
  return angle_;
}

} // namespace packet::parsing