#include "serverAgentEntityUpdateMovement.hpp"

#include "math/position.hpp"

#include <iostream>
#include <utility>

namespace packet::parsing {

ServerAgentEntityUpdateMovement::ServerAgentEntityUpdateMovement(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  globalId_ = stream.Read<uint32_t>();
  bool hasDestination_ = stream.Read<uint8_t>();
  if (hasDestination_) {
    packet::structures::Position tmpDestinationPosition;
    tmpDestinationPosition.regionId = stream.Read<uint16_t>();
    if (tmpDestinationPosition.isDungeon()) {
      // Dungeon
      tmpDestinationPosition.xOffset = stream.Read<int32_t>();
      tmpDestinationPosition.yOffset = stream.Read<int32_t>();
      tmpDestinationPosition.zOffset = stream.Read<int32_t>();
    } else {
      // World
      tmpDestinationPosition.xOffset = stream.Read<int16_t>();
      tmpDestinationPosition.yOffset = stream.Read<int16_t>();
      tmpDestinationPosition.zOffset = stream.Read<int16_t>();
    }
    math::position::normalize(tmpDestinationPosition);
    destinationPosition_.emplace(std::move(tmpDestinationPosition));
  } else {
    angleAction_ = static_cast<packet::enums::AngleAction>(stream.Read<uint8_t>());
    angle_ = stream.Read<uint16_t>();
  }
  bool hasSrc = stream.Read<uint8_t>();
  if (hasSrc) {
    packet::structures::Position tmpSourcePosition;
    tmpSourcePosition.regionId = stream.Read<uint16_t>();
    if (tmpSourcePosition.isDungeon()) {
      // Dungeon
      tmpSourcePosition.xOffset = stream.Read<int32_t>();
      uint32_t yOffsetBytes = stream.Read<uint32_t>();
      tmpSourcePosition.yOffset = *reinterpret_cast<float*>(&yOffsetBytes);
      tmpSourcePosition.zOffset = stream.Read<int32_t>();
    } else {
      // World
      tmpSourcePosition.xOffset = stream.Read<int16_t>();
      uint32_t yOffsetBytes = stream.Read<uint32_t>();
      tmpSourcePosition.yOffset = *reinterpret_cast<float*>(&yOffsetBytes);
      tmpSourcePosition.zOffset = stream.Read<int16_t>();
    }
    // Source position comes with X and Z values x10
    tmpSourcePosition.xOffset /= 10.0;
    tmpSourcePosition.zOffset /= 10.0;
    math::position::normalize(tmpSourcePosition);
    sourcePosition_.emplace(std::move(tmpSourcePosition));
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

const packet::structures::Position& ServerAgentEntityUpdateMovement::destinationPosition() const {
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

const packet::structures::Position& ServerAgentEntityUpdateMovement::sourcePosition() const {
  if (!sourcePosition_) {
    throw std::runtime_error("ServerAgentEntityUpdateMovement: sourcePosition does not exist");
  }
  return sourcePosition_.value();
}

} // namespace packet::parsing