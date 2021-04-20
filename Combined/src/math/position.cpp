#include "position.hpp"

#include <stdexcept>

namespace math::position {

float calculateDistance(const packet::structures::Position &srcPos, const packet::structures::Position &destPos) {
  // TODO: Move to common area
  if (srcPos.isDungeon() ^ destPos.isDungeon()) {
    throw std::runtime_error("Cannot calculate distances between different worlds");
  }
  if (srcPos.isDungeon()) {
    if (srcPos.dungeonId() != destPos.dungeonId()) {
      throw std::runtime_error("Cannot calculate distance between different dungeons");
    }
    // Simple distance calculation
    auto xDistance = destPos.xOffset - srcPos.xOffset;
    auto zDistance = destPos.zOffset - srcPos.zOffset;
    return sqrt(xDistance*xDistance + zDistance*zDistance);
  } else {
    // Need to account for regions
    constexpr int kRegionSize = 1920; // TODO: Move to a common area
    auto xDistance = (destPos.xSector() - srcPos.xSector()) * kRegionSize + (destPos.xOffset - srcPos.xOffset);
    auto zDistance = (destPos.zSector() - srcPos.zSector()) * kRegionSize + (destPos.zOffset - srcPos.zOffset);
    return sqrt(xDistance*xDistance + zDistance*zDistance);
  }
}

void normalize(packet::structures::Position &position) {
  if (position.isDungeon()) {
    // Nothing to normalize, only one "region"
    return;
  }
  constexpr int kRegionSize = 1920; // TODO: Move to a common area
  uint16_t newXSector = position.xSector() + position.xOffset/kRegionSize;
  uint16_t newZSector = position.zSector() + position.zOffset/kRegionSize;
  position.xOffset = std::fmod(position.xOffset, kRegionSize);
  position.zOffset = std::fmod(position.zOffset, kRegionSize);
  if (position.xOffset < 0) {
    position.xOffset += kRegionSize;
  }
  if (position.zOffset < 0) {
    position.zOffset += kRegionSize;
  }
  position.regionId = packet::structures::createWorldRegionId(newXSector, newZSector);
}

packet::structures::Position interpolateBetweenPoints(const packet::structures::Position &srcPos, const packet::structures::Position &destPos, float percent) {
  // TODO: Move to common area
  if (srcPos.isDungeon() ^ destPos.isDungeon()) {
    throw std::runtime_error("Cannot calculate distances between different worlds");
  }
  if (srcPos.isDungeon()) {
    if (srcPos.dungeonId() != destPos.dungeonId()) {
      throw std::runtime_error("Cannot calculate distance between different dungeons");
    }
    // Simple distance calculation
    packet::structures::Position newPosition = srcPos;
    auto xDistance = destPos.xOffset - srcPos.xOffset;
    auto yDistance = destPos.yOffset - srcPos.yOffset; // TODO: Do we care?
    auto zDistance = destPos.zOffset - srcPos.zOffset;
    newPosition.xOffset += xDistance*percent;
    newPosition.yOffset += yDistance*percent; // TODO: Do we care?
    newPosition.zOffset += zDistance*percent;
    return newPosition;
  } else {
    // Need to account for regions
    constexpr int kRegionSize = 1920; // TODO: Move to a common area
    packet::structures::Position newPosition = srcPos;
    auto xDistance = (destPos.xSector() - srcPos.xSector()) * kRegionSize + (destPos.xOffset - srcPos.xOffset);
    auto yDistance = destPos.yOffset - srcPos.yOffset; // TODO: Do we care?
    auto zDistance = (destPos.zSector() - srcPos.zSector()) * kRegionSize + (destPos.zOffset - srcPos.zOffset);
    newPosition.xOffset += xDistance*percent;
    newPosition.yOffset += yDistance*percent; // TODO: Do we care?
    newPosition.zOffset += zDistance*percent;
    normalize(newPosition);
    return newPosition;
  }
}

packet::structures::Position offset(const packet::structures::Position &srcPos, float xOffset, float yOffset) {
  packet::structures::Position newPosition = srcPos;
  newPosition.xOffset += xOffset;
  newPosition.zOffset += yOffset;
  normalize(newPosition);
  return newPosition;
}

} // namespace math::position