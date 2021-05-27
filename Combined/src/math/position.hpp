#ifndef MATH_POSITION_HPP_
#define MATH_POSITION_HPP_

#include "../packet/structures/packetInnerStructures.hpp"

#include <utility>

namespace math {

constexpr double kPi = 3.141592653589793;

} // namespace math

namespace math::position {

float calculateDistance(const packet::structures::Position &srcPos, const packet::structures::Position &destPos);
void normalize(packet::structures::Position &position);
packet::structures::Position interpolateBetweenPoints(const packet::structures::Position &srcPos, const packet::structures::Position &destPos, float percent);
packet::structures::Position offset(const packet::structures::Position &srcPos, float xOffset, float zOffset);

/**
 * Builds a region ID for the world from a regionX and regionY
 *
 * @param regionX The X coordinate of the region
 * @param regionY The Y coordinate of the region
 * @return A region ID for the world (not a dungeon) based on the given X,Y
 */
uint16_t worldRegionIdFromXY(const int regionX, const int regionY);

/**
 * Returns the x and y coordinates of the region for thee given region ID
 *
 * @param regionId The ID of the region
 * @return The x and y cooridnates of the given region
 */
std::pair<int,int> regionXYFromRegionId(const uint16_t regionId);

} // namespace math::position

#endif // MATH_POSITION_HPP_