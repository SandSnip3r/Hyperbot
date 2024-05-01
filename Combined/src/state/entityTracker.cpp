#include "entityTracker.hpp"
#include "event/event.hpp"
#include "helpers.hpp"

namespace state {

void EntityTracker::trackEntity(std::shared_ptr<entity::Entity> entity) {
  std::unique_lock<std::mutex> entityLockGuard(entityMutex_);
  if (entityMap_.find(entity->globalId) != entityMap_.end()) {
    throw std::runtime_error("EntityTracker::trackEntity Entity "+std::to_string(entity->globalId)+" already exists");
  }
  entityMap_.emplace(entity->globalId, entity);
}

void EntityTracker::stopTrackingEntity(sro::scalar_types::EntityGlobalId globalId) {
  std::unique_lock<std::mutex> entityLockGuard(entityMutex_);
  auto it = entityMap_.find(globalId);
  if (it != entityMap_.end()) {
    entityMap_.erase(it);
  } else {
    throw std::runtime_error("EntityTracker::stopTrackingEntity Entity "+std::to_string(globalId)+" does not exist");
  }
}

bool EntityTracker::trackingEntity(sro::scalar_types::EntityGlobalId globalId) const {
  std::unique_lock<std::mutex> entityLockGuard(entityMutex_);
  return (entityMap_.find(globalId) != entityMap_.end());
}

entity::Entity* EntityTracker::getEntity(sro::scalar_types::EntityGlobalId globalId) {
  std::unique_lock<std::mutex> entityLockGuard(entityMutex_);
  return privateCheckedGetEntity(globalId);
}

const entity::Entity* EntityTracker::getEntity(sro::scalar_types::EntityGlobalId globalId) const {
  std::unique_lock<std::mutex> entityLockGuard(entityMutex_);
  return privateCheckedGetEntity(globalId);
}

size_t EntityTracker::size() const {
  return entityMap_.size();
}

const std::map<sro::scalar_types::EntityGlobalId, std::shared_ptr<entity::Entity>>& EntityTracker::getEntityMap() const {
  return entityMap_;
}

entity::Entity* EntityTracker::privateCheckedGetEntity(sro::scalar_types::EntityGlobalId globalId) {
  auto it = entityMap_.find(globalId);
  if (it == entityMap_.end()) {
    throw std::runtime_error("Entity "+std::to_string(globalId)+" does not exist");
  }
  if (!it->second) {
    throw std::runtime_error("Entity "+std::to_string(globalId)+" is null");

  }
  return it->second.get();
}

const entity::Entity* EntityTracker::privateCheckedGetEntity(sro::scalar_types::EntityGlobalId globalId) const {
  auto it = entityMap_.find(globalId);
  if (it == entityMap_.end()) {
    throw std::runtime_error("Entity "+std::to_string(globalId)+" does not exist");
  }
  if (!it->second) {
    throw std::runtime_error("Entity "+std::to_string(globalId)+" is null");

  }
  return it->second.get();
}

} // namespace state