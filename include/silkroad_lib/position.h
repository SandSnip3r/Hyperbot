#ifndef POSITION_H_
#define POSITION_H_

#include "scalar_types.h"

#include <cstdint>

namespace sro {

using RegionId = uint16_t;
using DungeonId = uint16_t;
using Sector = uint8_t;

struct GameCoordinate {
  int x, y;
};

class Position {
public:
  // Position() = default;
  Position(RegionId regionId, float xOffset, float yOffset, float zOffset);
  RegionId regionId() const;
  bool isDungeon() const;
  DungeonId dungeonId() const;
  Sector xSector() const;
  Sector zSector() const;
  float xOffset() const;
  float yOffset() const;
  float zOffset() const;
  GameCoordinate toGameCoordinate() const;
private:
  RegionId regionId_;
  float xOffset_;
  float yOffset_;
  float zOffset_;

  void normalize();
};

} // namespace sro

#endif // POSITION_H_