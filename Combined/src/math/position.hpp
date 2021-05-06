#ifndef MATH_POSITION_HPP_
#define MATH_POSITION_HPP_

#include "../packet/structures/packetInnerStructures.hpp"

namespace math {

const double kPi = 3.141592653589793;

} // namespace math

namespace math::position {

float calculateDistance(const packet::structures::Position &srcPos, const packet::structures::Position &destPos);
void normalize(packet::structures::Position &position);
packet::structures::Position interpolateBetweenPoints(const packet::structures::Position &srcPos, const packet::structures::Position &destPos, float percent);
packet::structures::Position offset(const packet::structures::Position &srcPos, float xOffset, float zOffset);

} // namespace math::position

#endif // MATH_POSITION_HPP_