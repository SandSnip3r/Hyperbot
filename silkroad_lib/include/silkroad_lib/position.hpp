#ifndef SRO_POSITION_H_
#define SRO_POSITION_H_

#include "scalar_types.hpp"

#include <cstdint>
#include <ostream>

namespace sro {

using RegionId = uint16_t;
using DungeonId = uint16_t;
using Sector = uint8_t;
using Angle = uint16_t;

struct GameCoordinate {
  int x, y;
};

class Position {
public:
  Position() = default;
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
  RegionId regionId_{0};
  float xOffset_{0};
  float yOffset_{0};
  float zOffset_{0};

  void normalize();
};

std::ostream& operator<<(std::ostream &stream, const Position &pos);
bool operator==(const Position &pos1, const Position &pos2);
bool operator!=(const Position &pos1, const Position &pos2);

} // namespace sro

#endif // SRO_POSITION_H_