#include "constants.h"
#include "game_constants.h"
#include "position_math.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace sro::position_math {

float calculateDistance2D(const Position &srcPos, const Position &destPos) {
  using sro::game_constants::kRegionSize;
  if (srcPos.isDungeon() ^ destPos.isDungeon()) {
    throw std::runtime_error("Cannot calculate distances between different worlds");
  }
  if (srcPos.isDungeon()) {
    if (srcPos.dungeonId() != destPos.dungeonId()) {
      throw std::runtime_error("Cannot calculate distance between different dungeons");
    }
    // Simple distance calculation
    auto xDistance = destPos.xOffset() - srcPos.xOffset();
    auto zDistance = destPos.zOffset() - srcPos.zOffset();
    return sqrt(xDistance*xDistance + zDistance*zDistance);
  } else {
    // Need to account for regions
    auto xDistance = (destPos.xSector() - srcPos.xSector()) * kRegionSize + (destPos.xOffset() - srcPos.xOffset());
    auto zDistance = (destPos.zSector() - srcPos.zSector()) * kRegionSize + (destPos.zOffset() - srcPos.zOffset());
    return sqrt(xDistance*xDistance + zDistance*zDistance);
  }
}

Position interpolateBetweenPoints(const Position &srcPos, const Position &destPos, float percent) {
  using sro::game_constants::kRegionSize;
  if (srcPos.isDungeon() ^ destPos.isDungeon()) {
    throw std::runtime_error("Cannot interpolate between different worlds");
  }
  if (srcPos.isDungeon()) {
    if (srcPos.dungeonId() != destPos.dungeonId()) {
      throw std::runtime_error("Cannot interpolate between different dungeons");
    }
    // Simple distance calculation
    const auto xDistance = destPos.xOffset() - srcPos.xOffset();
    const auto yDistance = destPos.yOffset() - srcPos.yOffset();
    const auto zDistance = destPos.zOffset() - srcPos.zOffset();
    const auto newPositionXOffset = srcPos.xOffset() + xDistance*percent;
    const auto newPositionYOffset = srcPos.yOffset() + yDistance*percent;
    const auto newPositionZOffset = srcPos.zOffset() + zDistance*percent;
    return { srcPos.regionId(), newPositionXOffset, newPositionYOffset, newPositionZOffset };
  } else {
    // Need to account for regions
    auto xDistance = (destPos.xSector() - srcPos.xSector()) * kRegionSize + (destPos.xOffset() - srcPos.xOffset());
    auto yDistance = destPos.yOffset() - srcPos.yOffset();
    auto zDistance = (destPos.zSector() - srcPos.zSector()) * kRegionSize + (destPos.zOffset() - srcPos.zOffset());
    const auto newPositionXOffset = srcPos.xOffset() + xDistance*percent;
    const auto newPositionYOffset = srcPos.yOffset() + yDistance*percent;
    const auto newPositionZOffset = srcPos.zOffset() + zDistance*percent;
    return { srcPos.regionId(), newPositionXOffset, newPositionYOffset, newPositionZOffset };
  }
}

Position getNewPositionGivenAngleAndDistance(const Position &srcPos, MovementAngle angle, float distance) {
  const auto angleRadians = constants::k2Pi * static_cast<double>(angle) / std::numeric_limits<MovementAngle>::max();
  const float dx = distance * cos(angleRadians);
  const float dz = distance * sin(angleRadians);
  return { srcPos.regionId(),
           srcPos.xOffset() + dx,
           srcPos.yOffset(),
           srcPos.zOffset() + dz };
}

RegionId worldRegionIdFromSectors(const Sector xSector, const Sector zSector) {
  return ((xSector & 0xFF) | (static_cast<RegionId>(zSector & 0xFF) << 8));
}

std::pair<Sector,Sector> sectorsFromWorldRegionId(const RegionId regionId) {
  return {(regionId & 0xFF), ((regionId >> 8) & 0xFF)};
}

bool regionIsDungeon(const RegionId regionId) {
  return regionId & 0x8000;
}

} // namespace sro::position_math