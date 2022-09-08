#ifndef SRO_POSITION_MATH_H_
#define SRO_POSITION_MATH_H_

#include "position.h"

#include <utility>

namespace sro::position_math {

float calculateDistance2D(const Position &srcPos, const Position &destPos);
Position interpolateBetweenPoints(const Position &srcPos, const Position &destPos, float percent);
Position getNewPositionGivenAngleAndDistance(const Position &srcPos, MovementAngle angle, float distance);
Position createNewPositionWith2dOffset(const Position &startingPos, const float xOffset, const float zOffset);
RegionId worldRegionIdFromSectors(const Sector xSector, const Sector zSector);
std::pair<Sector,Sector> sectorsFromWorldRegionId(const RegionId regionId);
bool regionIsDungeon(const RegionId regionId);

} // namespace sro::position_math

#endif // SRO_POSITION_MATH_H_