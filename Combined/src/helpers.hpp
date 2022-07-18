#ifndef HELPERS_HPP_
#define HELPERS_HPP_

#include "packet/structures/packetInnerStructures.hpp"

float secondsToTravel(const packet::structures::Position &srcPosition, const packet::structures::Position &destPosition, const float currentSpeed);

#endif // HELPERS_HPP_