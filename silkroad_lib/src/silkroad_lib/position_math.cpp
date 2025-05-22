#include <silkroad_lib/constants.hpp>
#include <silkroad_lib/game_constants.hpp>
#include <silkroad_lib/position_math.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace sro::position_math {

std::pair<float,float> calculateOffset2d(const Position &srcPos, const Position &destPos) {
  using sro::game_constants::kRegionSize;
  if (srcPos.isDungeon() ^ destPos.isDungeon()) {
    throw std::runtime_error("Cannot calculate offset between different worlds");
  }
  float dx, dz;
  if (srcPos.isDungeon()) {
    if (srcPos.dungeonId() != destPos.dungeonId()) {
      throw std::runtime_error("Cannot calculate offset between different dungeons");
    }
    // Simple distance calculation
    dx = destPos.xOffset() - srcPos.xOffset();
    dz = destPos.zOffset() - srcPos.zOffset();
  } else {
    // Need to account for regions
    dx = (destPos.xSector() - srcPos.xSector()) * kRegionSize + (destPos.xOffset() - srcPos.xOffset());
    dz = (destPos.zSector() - srcPos.zSector()) * kRegionSize + (destPos.zOffset() - srcPos.zOffset());
  }
  return {dx, dz};
}
Angle calculateAngleOfLine(const Position &srcPos, const Position &destPos) {
  const auto [dx, dz] = calculateOffset2d(srcPos, destPos);
  double angle = std::atan(dz/dx);
  if (dx < 0) {
    angle += constants::kPi;
  } else if (dz < 0) {
    angle += constants::k2Pi;
  }
  return std::round((angle / constants::k2Pi) * std::numeric_limits<Angle>::max());
}

float calculateDistance2d(const Position &srcPos, const Position &destPos) {
  const auto [dx, dz] = calculateOffset2d(srcPos, destPos);
  return sqrt(dx*dx + dz*dz);
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

Position getNewPositionGivenAngleAndDistance(const Position &srcPos, Angle angle, float distance) {
  const auto angleRadians = constants::k2Pi * static_cast<double>(angle) / std::numeric_limits<Angle>::max();
  const float dx = distance * cos(angleRadians);
  const float dz = distance * sin(angleRadians);
  return { srcPos.regionId(),
           srcPos.xOffset() + dx,
           srcPos.yOffset(),
           srcPos.zOffset() + dz };
}

Position createNewPositionWith2dOffset(const Position &startingPos, const float xOffset, const float zOffset) {
  return {startingPos.regionId(), startingPos.xOffset()+xOffset, startingPos.yOffset(), startingPos.zOffset()+zOffset};
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

bool pointIsInRect2d(const Position &point, const Position &rectStart, const Position &rectEnd) {
  // Normalize everything to `point`'s region
  const auto [rectStartX, rectStartZ] = calculateOffsetInOtherRegion(rectStart, point);
  const auto [rectEndX, rectEndZ] = calculateOffsetInOtherRegion(rectEnd, point);
  return point.xOffset() >= rectStartX &&
         point.zOffset() >= rectStartZ &&
         point.xOffset() <= rectEndX &&
         point.zOffset() <= rectEndZ;
}

std::pair<float,float> calculateOffsetInOtherRegion(const Position &point, const Position &other) {
  if (point.isDungeon() != other.isDungeon()) {
    throw std::runtime_error("Cannot calculate offset between point in dungeon and point not in dungeon");
  }
  return { point.xOffset() + sro::game_constants::kRegionWidth * (static_cast<int>(point.xSector()) - other.xSector()),
           point.zOffset() + sro::game_constants::kRegionHeight * (static_cast<int>(point.zSector()) - other.zSector()) };
}

} // namespace sro::position_math