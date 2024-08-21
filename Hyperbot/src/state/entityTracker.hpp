#ifndef STATE_ENTITY_TRACKER_HPP_
#define STATE_ENTITY_TRACKER_HPP_

#include "broker/eventBroker.hpp"
#include "entity/entity.hpp"
#include "packet/parsing/parsedPacket.hpp"

#include <absl/container/flat_hash_map.h>

#include <memory>
#include <mutex>
#include <type_traits>
#include <vector>

namespace state {
  
class EntityTracker {
public:
  // When an entity spawns, call this to hold on to the entity. Since multiple characters can see the spawning of a single entity, if the entity is already tracked, a "reference count" is incremented.
  // Returns `true` if this tracks a new entity, `false` if this entity is already being tracked.
  bool entitySpawned(std::shared_ptr<entity::Entity> entity, broker::EventBroker &eventBroker);
  // When an entity despawns, call this to release the entity. Since multiple characters can see the despawning of a single entity, a "reference count" is decremented. If that count hits 0, the entity is deleted.
  void entityDespawned(sro::scalar_types::EntityGlobalId globalId, broker::EventBroker &eventBroker);

  bool trackingEntity(sro::scalar_types::EntityGlobalId globalId) const;

  template<typename EntityType = entity::Entity>
  std::shared_ptr<EntityType> getEntity(sro::scalar_types::EntityGlobalId globalId) const {
    std::unique_lock<std::mutex> entityMapLockGuard(entityMapMutex_);
    auto it = entityMap_.find(globalId);
    if (it == entityMap_.end()) {
      throw std::runtime_error(absl::StrFormat("EntityTracker::getEntity; Entity ID %d does not exist", globalId));
    }
    if (!it->second) {
      throw std::runtime_error(absl::StrFormat("EntityTracker::getEntity; Entity ID %d is null", globalId));
    }
    if constexpr (std::is_same_v<EntityType, entity::Entity>) {
      return it->second;
    } else {
      return std::dynamic_pointer_cast<EntityType>(it->second);
    }
  }

  size_t size() const;
  const absl::flat_hash_map<sro::scalar_types::EntityGlobalId, std::shared_ptr<entity::Entity>>& getEntityMap() const; // TODO: Remove
private:
  // Guards `entityMap_` and `entityReferenceCountMap_`.
  mutable std::mutex entityMapMutex_;

  absl::flat_hash_map<sro::scalar_types::EntityGlobalId, std::shared_ptr<entity::Entity>> entityMap_;
  absl::flat_hash_map<sro::scalar_types::EntityGlobalId, int> entityReferenceCountMap_;
};

} // namespace state

#endif // STATE_ENTITY_TRACKER_HPP_