#ifndef ENTITY_DATA_SELF_HPP_
#define ENTITY_DATA_SELF_HPP_

#include "common.hpp"
#include "entity.hpp"

#include <silkroad_lib/position.hpp>

#include <cstdint>
#include <optional>

namespace entity_data {

class Self : public Character {
public:
  virtual ~Self() = default;
  int64_t expRequired;
  uint32_t currentHp, currentMp;
  std::optional<uint32_t> maxHp, maxMp;
  static const constexpr int32_t spExpRequired{400};
};

} // namespace entity_data

#endif // ENTITY_DATA_SELF_HPP_
