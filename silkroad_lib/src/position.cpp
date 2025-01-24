#include "game_constants.hpp"
#include "position.hpp"
#include "position_math.hpp"

#include <absl/strings/str_format.h>

#include <cmath>

namespace sro {

Position::Position(RegionId regionId, float xOffset, float yOffset, float zOffset) : regionId_(regionId), xOffset_(xOffset), yOffset_(yOffset), zOffset_(zOffset) {
  normalize();
}

RegionId Position::regionId() const {
  return regionId_;
}

bool Position::isDungeon() const {
  return position_math::regionIsDungeon(regionId_);
}

DungeonId Position::dungeonId() const {
  return regionId_ & 0xFF;
}

Sector Position::xSector() const {
  return regionId_ & 0xFF;
}

Sector Position::zSector() const {
  return (regionId_ >> 8) & 0x7F;
}

float Position::xOffset() const {
  return xOffset_;
}

float Position::yOffset() const {
  return yOffset_;
}

float Position::zOffset() const {
  return zOffset_;
}

GameCoordinate Position::toGameCoordinate() const {
  using sro::game_constants::kRegionSize;
  return { static_cast<int>((xSector() - 135) * kRegionSize/10.0 + xOffset() / 10.0),
           static_cast<int>((zSector() - 92) * kRegionSize/10.0 + zOffset() / 10.0) };
}

std::string Position::toString() const {
  std::string regionDetails;
  if (isDungeon()) {
    regionDetails = absl::StrFormat("d-%d", dungeonId());
  } else {
    regionDetails = absl::StrFormat("%d,%d", xSector(), zSector());
  }
  return absl::StrFormat("{%d(%s) %.3f,%.3f,%.3f}", regionId_, regionDetails, xOffset_, yOffset_, zOffset_);
}

void Position::normalize() {
  using sro::game_constants::kRegionSize;
  if (isDungeon()) {
    // Nothing to normalize, only one "region"
    return;
  }
  Sector newXSector = xSector() + static_cast<Sector>(floor(xOffset_/kRegionSize));
  Sector newZSector = zSector() + static_cast<Sector>(floor(zOffset_/kRegionSize));
  xOffset_ = std::fmod(xOffset_, kRegionSize);
  zOffset_ = std::fmod(zOffset_, kRegionSize);
  if (xOffset_ < 0) {
    xOffset_ += kRegionSize;
  }
  if (zOffset_ < 0) {
    zOffset_ += kRegionSize;
  }
  regionId_ = sro::position_math::worldRegionIdFromSectors(newXSector, newZSector);
}

std::ostream& operator<<(std::ostream &stream, const Position &pos) {
  stream << '{';
  if (pos.isDungeon()) {
    stream << (int)pos.dungeonId();
  } else {
    stream << (int)pos.xSector() << ',' << (int)pos.zSector();
  }
  stream << " (" << pos.xOffset() << ',' << pos.yOffset() << ',' << pos.zOffset() << ")}";
  return stream;
}

bool operator==(const Position &pos1, const Position &pos2) {
  if (pos1.regionId() != pos2.regionId()) {
    return false;
  }
  return ((pos1.xOffset() == pos2.xOffset()) &&
          (pos1.yOffset() == pos2.yOffset()) &&
          (pos1.zOffset() == pos2.zOffset()));
}

bool operator!=(const Position &pos1, const Position &pos2) {
  return !(pos1 == pos2);
}

} // namespace sro