#include "entityTracker.hpp"
#include "event/event.hpp"
#include "helpers.hpp"

#include <absl/strings/str_format.h>

namespace state {

void EntityTracker::trackEntity(std::shared_ptr<entity::Entity> entity) {
  std::unique_lock<std::mutex> entityMapLockGuard(entityMapMutex_);
  if (entityMap_.find(entity->globalId) != entityMap_.end()) {
    throw std::runtime_error(absl::StrFormat("EntityTracker::trackEntity; Entity ID %d already exists", entity->globalId));
  }
  entityMap_.emplace(entity->globalId, entity);
}

void EntityTracker::stopTrackingEntity(sro::scalar_types::EntityGlobalId globalId) {
  std::unique_lock<std::mutex> entityMapLockGuard(entityMapMutex_);
  auto it = entityMap_.find(globalId);
  if (it != entityMap_.end()) {
    entityMap_.erase(it);
  } else {
    throw std::runtime_error(absl::StrFormat("EntityTracker::stopTrackingEntity; Entity ID %d does not exist", globalId));
  }
}

bool EntityTracker::trackingEntity(sro::scalar_types::EntityGlobalId globalId) const {
  std::unique_lock<std::mutex> entityMapLockGuard(entityMapMutex_);
  return (entityMap_.find(globalId) != entityMap_.end());
}

// std::shared_ptr<entity::Entity> EntityTracker::getEntity(sro::scalar_types::EntityGlobalId globalId) const {
//   std::unique_lock<std::mutex> entityMapLockGuard(entityMapMutex_);
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
  std::unique_lock<std::mutex> entityMapLockGuard(entityMapMutex_);
  return entityMap_.size();
}

const std::map<sro::scalar_types::EntityGlobalId, std::shared_ptr<entity::Entity>>& EntityTracker::getEntityMap() const {
  return entityMap_;
}

} // namespace state