#ifndef CHARACTER_DATA_HPP_
#define CHARACTER_DATA_HPP_

#include <silkroad_lib/position.h>

#include <chrono>
#include <cstdint>
#include <optional>
#include <variant>

struct Movement {
  float speed;
  std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
  sro::Position srcPos;
  enum MovementType {
    kToDestination,
    kTowardAngle
  };
  std::variant<sro::Position, uint16_t> destPosOrAngle;
};

class CharacterData {
public:
  int64_t expRequired;
  uint32_t currentHp, currentMp;
  std::optional<uint32_t> maxHp, maxMp;
  std::optional<Movement> movement;
  static const constexpr int32_t spExpRequired{400};
};

#endif // CHARACTER_DATA_HPP_
