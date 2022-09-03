#ifndef POSITION_MATH_H_
#define POSITION_MATH_H_

#include "position.h"

#include <pair>

namespace sro::position_math {

float calculateDistance2D(const Position &srcPos, const Position &destPos);
Position interpolateBetweenPoints(const Position &srcPos, const Position &destPos, float percent);
Position getNewPositionGivenAngleAndDistance(const Position &srcPos, MovementAngle angle, float distance);
RegionId worldRegionIdFromSectors(const Sector xSector, const Sector zSector);
std::pair<Sector,Sector> sectorsFromWorldRegionId(const RegionId regionId);

} // namespace sro::position_math

#endif // POSITION_MATH_H_