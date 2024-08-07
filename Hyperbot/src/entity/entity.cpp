#include "entity.hpp"
#include "helpers.hpp"

#include <silkroad_lib/constants.h>
#include <silkroad_lib/position_math.h>

#include <absl/log/log.h>

namespace entity {

void Entity::initializePosition(const sro::Position &position) {
  position_ = position;
}

void Entity::initializeAngle(sro::Angle angle) {
  angle_ = angle;
}

void Entity::initializeEventBroker(broker::EventBroker &eventBroker) {
  eventBroker_ = &eventBroker;
}

sro::Position Entity::position() const {
  return position_;
}

sro::Angle Entity::angle() const {
  return angle_;
}

} // namespace entity