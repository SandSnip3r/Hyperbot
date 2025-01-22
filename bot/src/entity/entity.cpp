#include "entity.hpp"
#include "helpers.hpp"

#include <silkroad_lib/constants.hpp>
#include <silkroad_lib/position_math.hpp>

#include <absl/log/log.h>
#include <absl/strings/str_format.h>

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

std::string Entity::toStringImpl(const pk2::GameData *gameData) const {
  return absl::StrFormat("entity %d", globalId);
}

std::string_view toString(EntityType entityType) {
  if (entityType == EntityType::kSelf) {
    return "Self";
  } else if (entityType == EntityType::kCharacter) {
    return "Character";
  } else if (entityType == EntityType::kPlayerCharacter) {
    return "PlayerCharacter";
  } else if (entityType == EntityType::kNonplayerCharacter) {
    return "NonplayerCharacter";
  } else if (entityType == EntityType::kMonster) {
    return "Monster";
  } else if (entityType == EntityType::kItem) {
    return "Item";
  } else if (entityType == EntityType::kPortal) {
    return "Portal";
  }
  throw std::runtime_error("Invalid EntityType");
}

} // namespace entity