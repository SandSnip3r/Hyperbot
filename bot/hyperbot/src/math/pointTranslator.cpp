#include "pointTranslator.hpp"

namespace math {

PointTranslator::PointTranslator(const sro::Position &referenceFrameCenter) : referenceFrameCenter_(referenceFrameCenter) {
  //
}

pathfinder::Vector PointTranslator::getReferencePathfinderPoint() {
  return {0.0,0.0};
}

pathfinder::Vector PointTranslator::sroToPathfinder(const sro::Position &point) const {
  const auto [dx, dy] = sro::position_math::calculateOffset2d(referenceFrameCenter_, point);
  return {dx, dy};
}

sro::Position PointTranslator::pathfinderToSro(const pathfinder::Vector &point) const {
  return sro::position_math::createNewPositionWith2dOffset(referenceFrameCenter_, point.x(), point.y());
}

} // namespace math