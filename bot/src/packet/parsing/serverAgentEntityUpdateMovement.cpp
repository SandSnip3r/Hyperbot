#include "serverAgentEntityUpdateMovement.hpp"

#include <silkroad_lib/position_math.hpp>

#include <utility>

namespace packet::parsing {

ServerAgentEntityUpdateMovement::ServerAgentEntityUpdateMovement(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  globalId_ = stream.Read<uint32_t>();
  bool hasDestination_ = stream.Read<uint8_t>();
  if (hasDestination_) {
    sro::RegionId regionId = stream.Read<sro::RegionId>();
    int32_t xOffset, yOffset, zOffset;
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
  bool hasSrc = stream.Read<uint8_t>();
  if (hasSrc) {
    sro::RegionId regionId = stream.Read<sro::RegionId>();
    int32_t xOffset, zOffset;
    float yOffset;
    if (sro::position_math::regionIsDungeon(regionId)) {
      // Dungeon
      xOffset = stream.Read<int32_t>();
      yOffset = stream.Read<float>();
      zOffset = stream.Read<int32_t>();
    } else {
      // World
      xOffset = stream.Read<int16_t>();
      yOffset = stream.Read<float>();
      zOffset = stream.Read<int16_t>();
    }
    // Source position comes with X and Z values x10
    float xOffsetF = xOffset / 10.0;
    float zOffsetF = zOffset / 10.0;
    sourcePosition_.emplace(regionId, xOffsetF, yOffset, zOffsetF);
  }
}

uint32_t ServerAgentEntityUpdateMovement::globalId() const {
  return globalId_;
}

bool ServerAgentEntityUpdateMovement::hasDestination() const {
  return destinationPosition_.has_value();
}

bool ServerAgentEntityUpdateMovement::hasSource() const {
  return sourcePosition_.has_value();
}

const sro::Position& ServerAgentEntityUpdateMovement::destinationPosition() const {
  if (!destinationPosition_) {
    throw std::runtime_error("ServerAgentEntityUpdateMovement: destinationPosition does not exist");
  }
  return destinationPosition_.value();
}

packet::enums::AngleAction ServerAgentEntityUpdateMovement::angleAction() const {
  return angleAction_;
}

uint16_t ServerAgentEntityUpdateMovement::angle() const {
  return angle_;
}

const sro::Position& ServerAgentEntityUpdateMovement::sourcePosition() const {
  if (!sourcePosition_) {
    throw std::runtime_error("ServerAgentEntityUpdateMovement: sourcePosition does not exist");
  }
  return sourcePosition_.value();
}

} // namespace packet::parsing