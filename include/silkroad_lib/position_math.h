#ifndef SRO_POSITION_MATH_H_
#define SRO_POSITION_MATH_H_

#include "position.h"

#include <utility>

namespace sro::position_math {

std::pair<float,float> calculateOffset2d(const Position &srcPos, const Position &destPos);
Angle calculateAngleOfLine(const Position &srcPos, const Position &destPos);
float calculateDistance2d(const Position &srcPos, const Position &destPos);
Position interpolateBetweenPoints(const Position &srcPos, const Position &destPos, float percent);
Position getNewPositionGivenAngleAndDistance(const Position &srcPos, Angle angle, float distance);
Position createNewPositionWith2dOffset(const Position &startingPos, const float xOffset, const float zOffset);
RegionId worldRegionIdFromSectors(const Sector xSector, const Sector zSector);
std::pair<Sector,Sector> sectorsFromWorldRegionId(const RegionId regionId);
bool regionIsDungeon(const RegionId regionId);

} // namespace sro::position_math

#endif // SRO_POSITION_MATH_H_