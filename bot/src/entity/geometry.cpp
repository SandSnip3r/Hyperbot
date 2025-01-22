#include "geometry.hpp"
#include "math/pointTranslator.hpp"

// From Pathfinder
#include "math_helpers.h"
#include "vector.h"

#include <silkroad_lib/position_math.hpp>

namespace entity {

Circle::Circle(const sro::Position &center, double radius) : center_(center), radius_(radius) {}

std::optional<double> Circle::timeUntilEnter(const sro::Position &srcPos, const sro::Position &destPos, float movementSpeed) const {
  if (pointIsInside(srcPos)) {
    // Already inside, cannot enter
    return {};
  }
  const math::PointTranslator pointTranslator(center_);
  const auto [srcPointDx, srcPointDy] = sro::position_math::calculateOffset2d(center_, srcPos);
  const auto [destPointDx, destPointDy] = sro::position_math::calculateOffset2d(center_, destPos);
  pathfinder::Vector circleCenterPoint = pointTranslator.getReferencePathfinderPoint();
  pathfinder::Vector srcPoint = pointTranslator.sroToPathfinder(srcPos);
  pathfinder::Vector destPoint = pointTranslator.sroToPathfinder(destPos);
  pathfinder::Vector intersectionPoint;
  const int intersectionCount = pathfinder::math::lineSegmentIntersectsWithCircle(srcPoint, destPoint, circleCenterPoint, radius_, &intersectionPoint);
  if (intersectionCount == 0) {
    return {};
  }
  const auto destIsInsideCircle = pointIsInside(destPos);
  if (intersectionCount == 1 && !destIsInsideCircle) {
    // Only touching the boundary of the circle, will not enter
    return {};
  } else if (intersectionCount == 2 && destIsInsideCircle) {
    throw std::runtime_error("Cannot have two intersection points when one point is inside the circle");
  }
  // Even if there are 2 intersection points, we only care about the first one
  sro::Position intersectionPos = pointTranslator.pathfinderToSro(intersectionPoint);
  const float distance = sro::position_math::calculateDistance2d(srcPos, intersectionPos);
  return distance / movementSpeed;
}

std::optional<double> Circle::timeUntilEnter(const sro::Position &srcPos, sro::Angle movementAngle, float movementSpeed) const {
  if (pointIsInside(srcPos)) {
    // Already inside, cannot enter
    return {};
  }
  // TODO:
  return {};
}

std::optional<double> Circle::timeUntilExit(const sro::Position &srcPos, const sro::Position &destPos, float movementSpeed) const {
  if (!pointIsInside(srcPos)) {
    // Not inside; can't exit
    return {};
  }
  if (pointIsInside(destPos)) {
    // Destination is inside, will not exit
    return {};
  }
  const math::PointTranslator pointTranslator(center_);
  const auto [srcPointDx, srcPointDy] = sro::position_math::calculateOffset2d(center_, srcPos);
  const auto [destPointDx, destPointDy] = sro::position_math::calculateOffset2d(center_, destPos);
  pathfinder::Vector circleCenterPoint = pointTranslator.getReferencePathfinderPoint();
  pathfinder::Vector srcPoint = pointTranslator.sroToPathfinder(srcPos);
  pathfinder::Vector destPoint = pointTranslator.sroToPathfinder(destPos);
  pathfinder::Vector intersectionPoint;
  const int intersectionCount = pathfinder::math::lineSegmentIntersectsWithCircle(srcPoint, destPoint, circleCenterPoint, radius_, &intersectionPoint);
  if (intersectionCount != 1) {
    throw std::runtime_error("1 intersection point is the only possible result");
  }
  sro::Position intersectionPos = pointTranslator.pathfinderToSro(intersectionPoint);
  const float distance = sro::position_math::calculateDistance2d(srcPos, intersectionPos);
  return distance / movementSpeed;
}

std::optional<double> Circle::timeUntilExit(const sro::Position &srcPos, sro::Angle movementAngle, float movementSpeed) const {
  if (!pointIsInside(srcPos)) {
    // Not inside; can't exit
    return {};
  }
  // TODO
  return {};
}

std::unique_ptr<Geometry> Circle::clone() const {
  return std::unique_ptr<Geometry>(new Circle(*this));
}

const sro::Position& Circle::center() const {
  return center_;
}

double Circle::radius() const {
  return radius_;
}

bool Circle::pointIsInside(const sro::Position &point) const {
  return sro::position_math::calculateDistance2d(center_, point) <= radius_;
}

// TODO: Rectangle implementation

} // namespace entity