#ifndef MATH_POINTTRANSLATOR_HPP_
#define MATH_POINTTRANSLATOR_HPP_

// From Pathfinder
#include "vector.h"

#include <silkroad_lib/position.hpp>
#include <silkroad_lib/position_math.hpp>

namespace math {

class PointTranslator {
public:
  PointTranslator(const sro::Position &referenceFrameCenter);
  static pathfinder::Vector getReferencePathfinderPoint();
  pathfinder::Vector sroToPathfinder(const sro::Position &point) const;
  sro::Position pathfinderToSro(const pathfinder::Vector &point) const;
private:
  const sro::Position referenceFrameCenter_;
};

} // namespace math

#endif // MATH_POINTTRANSLATOR_HPP_