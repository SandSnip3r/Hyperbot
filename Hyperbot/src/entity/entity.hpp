#ifndef ENTITY_ENTITY_HPP_
#define ENTITY_ENTITY_HPP_

#include "broker/eventBroker.hpp"

#include <silkroad_lib/position.h>
#include <silkroad_lib/scalar_types.h>

#include <cstdint>
#include <string_view>

namespace entity {

// TODO: This seems redundant when we can just dynamic_cast the entity. However, dynamic_casting will require a specific order if we care about whether something is not a base class of another thing.
enum class EntityType {
  kSelf,
  kCharacter,
  kPlayerCharacter,
  kNonplayerCharacter,
  kMonster,
  kItem,
  kPortal
};

class Entity {
public:
  sro::scalar_types::ReferenceObjectId refObjId;
  uint8_t typeId1, typeId2, typeId3, typeId4;
  sro::scalar_types::EntityGlobalId globalId;
  void initializePosition(const sro::Position &position);
  void initializeAngle(sro::Angle angle);
  virtual void initializeEventBroker(broker::EventBroker &eventBroker);
  virtual sro::Position position() const;
  sro::Angle angle() const;
  virtual ~Entity() = default;
  virtual EntityType entityType() const = 0;
protected:
  sro::Position position_;
  sro::Angle angle_;
  broker::EventBroker *eventBroker_{nullptr};
};

std::string_view toString(EntityType entityType);

} // namespace entity

#endif // ENTITY_ENTITY_HPP_