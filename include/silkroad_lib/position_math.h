#ifndef POSITION_MATH_H_
#define POSITION_MATH_H_

#include "position.h"

namespace sro::position_math {

float calculateDistance2D(const Position &srcPos, const Position &destPos);
Position interpolateBetweenPoints(const Position &srcPos, const Position &destPos, float percent);
RegionId worldRegionIdFromSectors(const Sector xSector, const Sector zSector);

} // namespace sro::position_math

#endif // POSITION_MATH_H_