#ifndef CHARACTER_DATA_HPP_
#define CHARACTER_DATA_HPP_

#include <cstdint>
#include <optional>

class CharacterData {
public:
  int64_t expRequired;
  uint32_t currentHp, currentMp;
  std::optional<uint32_t> maxHp, maxMp;
  static const constexpr int32_t spExpRequired{400};
};

#endif // CHARACTER_DATA_HPP_
