#ifndef ENTITY_DATA_COMMON_HPP_
#define ENTITY_DATA_COMMON_HPP_

#include <silkroad_lib/position.hpp>

#include <chrono>
#include <variant>

namespace entity_data {

struct Movement {
  float speed;
  std::chrono::time_point<std::chrono::steady_clock> startTime;
  sro::Position srcPos;
  enum MovementType {
    kToDestination,
    kTowardAngle
  };
  std::variant<sro::Position, uint16_t> destPosOrAngle;
};

} // namespace entity_data

#endif // ENTITY_DATA_COMMON_HPP_