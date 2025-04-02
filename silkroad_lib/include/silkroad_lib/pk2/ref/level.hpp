#ifndef PK2_REF_LEVEL_HPP_
#define PK2_REF_LEVEL_HPP_

#include <cstdint>
#include <string>
#include <ostream>

namespace sro::pk2::ref {

struct Level {
  uint8_t lvl;
  int64_t exp_C;
  int32_t exp_M;
  int32_t cost_M;
  int32_t cost_ST;
  int32_t gust_Mob_Exp;
  int32_t jobExp_Trader;
  int32_t jobExp_Robber;
  int32_t jobExp_Hunter;
};

std::ostream& operator<<(std::ostream &stream, const Level &level);

} // namespace sro::pk2::ref

#endif // PK2_REF_LEVEL_HPP_