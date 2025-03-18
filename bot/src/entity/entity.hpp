#ifndef ENTITY_ENTITY_HPP_
#define ENTITY_ENTITY_HPP_

#include "broker/eventBroker.hpp"

#include <silkroad_lib/position.hpp>
#include <silkroad_lib/scalar_types.hpp>

#include <cstdint>
#include <string>
#include <string_view>

namespace state {
class WorldState;
} // namespace state

namespace pk2 {
class GameData;
} // namespace pk2

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

  // TODO: For now, this function takes a reference to the WorldState, that is because we use WorldState::mutex to protect all data inside the WorldState. Entities are inside the WorldState. Entities can subscribe to events. When an entity receives an event, it needs to lock the WorldState::mutex.
  virtual void initializeEventBroker(broker::EventBroker &eventBroker, state::WorldState &worldState);
  virtual sro::Position position() const;
  sro::Angle angle() const;
  virtual ~Entity() = default;
  virtual EntityType entityType() const = 0;
  std::string toString() const { return toStringImpl(nullptr); }
  std::string toString(const pk2::GameData &gameData) const { return toStringImpl(&gameData); }
protected:
  sro::Position position_;
  sro::Angle angle_;
  broker::EventBroker *eventBroker_{nullptr};
  state::WorldState *worldState_{nullptr};
  virtual std::string toStringImpl(const pk2::GameData *gameData) const;
};

std::string_view toString(EntityType entityType);

} // namespace entity

#endif // ENTITY_ENTITY_HPP_