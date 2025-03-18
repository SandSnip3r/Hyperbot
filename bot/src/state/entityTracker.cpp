#include "entityTracker.hpp"
#include "event/event.hpp"
#include "helpers.hpp"
#include "state/worldState.hpp"

#include <absl/strings/str_format.h>

namespace state {

bool EntityTracker::entitySpawned(std::shared_ptr<entity::Entity> entity, broker::EventBroker &eventBroker, WorldState &worldState) {
  {
    std::unique_lock entityMapLockGuard(entityMapMutex_);
    int &entityReferenceCount = entityReferenceCountMap_[entity->globalId];
    ++entityReferenceCount;
    if (entityReferenceCount > 1) {
      auto it = entityMap_.find(entity->globalId);
      VLOG(2) << "Already tracking " << entity::toString(entity->entityType()) << " " << entity->toString() << " as a " << entity::toString(it->second->entityType());
      // TODO: Evaluate if this is a more descriptive version of that entity. I think the only case when this happens is if we want to promote a PlayerCharacter to a Self.
      // TODO: Update entity with any potentially new information in this entity.
      return false;
    }
    VLOG(1) << "Tracking new " << entity::toString(entity->entityType()) << " " << entity->toString();
    entity->initializeEventBroker(eventBroker, worldState);
    entityMap_.emplace(entity->globalId, entity);
  }
  eventBroker.publishEvent<event::EntitySpawned>(entity->globalId);
  return true;
}

bool EntityTracker::entityDespawned(sro::scalar_types::EntityGlobalId globalId, broker::EventBroker &eventBroker) {
  {
    std::unique_lock entityMapLockGuard(entityMapMutex_);
    int &entityReferenceCount = entityReferenceCountMap_[globalId];
    --entityReferenceCount;
    if (entityReferenceCount != 0) {
      auto it = entityMap_.find(globalId);
      VLOG(2) << "Not the last one to see " << entity::toString(it->second->entityType()) << " " << it->second->toString();
      // There are other characters who can still see this entity. Do not delete it.
      return false;
    }
    auto it = entityMap_.find(globalId);
    if (it != entityMap_.end()) {
      VLOG(1) << "Last one to see " << entity::toString(it->second->entityType()) << " " << it->second->toString() << ", deleting";
      entityMap_.erase(it);
    } else {
      throw std::runtime_error(absl::StrFormat("EntityTracker::entityDespawned; Entity ID %d does not exist", globalId));
    }
  }
  eventBroker.publishEvent<event::EntityDespawned>(globalId);
  return true;
}

bool EntityTracker::trackingEntity(sro::scalar_types::EntityGlobalId globalId) const {
  std::unique_lock entityMapLockGuard(entityMapMutex_);
  return (entityMap_.find(globalId) != entityMap_.end());
}

// std::shared_ptr<entity::Entity> EntityTracker::getEntity(sro::scalar_types::EntityGlobalId globalId) const {
//   std::unique_lock entityMapLockGuard(entityMapMutex_);
//   auto it = entityMap_.find(globalId);
//   if (it == entityMap_.end()) {
//     throw std::runtime_error(absl::StrFormat("EntityTracker::getEntity; Entity ID %d does not exist", globalId));
//   }
//   if (!it->second) {
//     throw std::runtime_error(absl::StrFormat("EntityTracker::getEntity; Entity ID %d is null", globalId));

//   }
//   return it->second;
// }

size_t EntityTracker::size() const {
  std::unique_lock entityMapLockGuard(entityMapMutex_);
  return entityMap_.size();
}

const absl::flat_hash_map<sro::scalar_types::EntityGlobalId, std::shared_ptr<entity::Entity>>& EntityTracker::getEntityMap() const {
  return entityMap_;
}

} // namespace state