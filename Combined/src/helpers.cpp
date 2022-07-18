#include "helpers.hpp"
#include "math/position.hpp"

float secondsToTravel(const packet::structures::Position &srcPosition, const packet::structures::Position &destPosition, const float currentSpeed) {
  auto distance = math::position::calculateDistance(srcPosition, destPosition);
  return distance / currentSpeed;
}