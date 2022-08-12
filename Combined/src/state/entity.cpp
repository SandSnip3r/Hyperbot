#include "entity.hpp"

namespace state {

void Entity::trackEntity(std::shared_ptr<packet::parsing::Object> obj) {
  std::unique_lock<std::mutex> entityLockGuard(entityMutex_);
  if (entityMap_.find(obj->gId) != entityMap_.end()) {
    throw std::runtime_error("Entity::trackEntity Entity "+std::to_string(obj->gId)+" already exists");
  }
  entityMap_.emplace(obj->gId, obj);
}

void Entity::stopTrackingEntity(EntityId gId) {
  std::unique_lock<std::mutex> entityLockGuard(entityMutex_);
  auto it = entityMap_.find(gId);
  if (it != entityMap_.end()) {
    entityMap_.erase(it);
  } else {
    throw std::runtime_error("Entity::stopTrackingEntity Entity "+std::to_string(gId)+" does not exist");
  }
}

bool Entity::trackingEntity(EntityId gId) const {
  std::unique_lock<std::mutex> entityLockGuard(entityMutex_);
  return (entityMap_.find(gId) != entityMap_.end());
}

packet::parsing::Object* Entity::getEntity(EntityId gId) {
  std::unique_lock<std::mutex> entityLockGuard(entityMutex_);
  auto it = entityMap_.find(gId);
  if (it == entityMap_.end()) {
    throw std::runtime_error("Entity::getEntity Enttiy "+std::to_string(gId)+" does not exist");
  }
  return it->second.get();
}

const packet::parsing::Object* Entity::getEntity(EntityId gId) const {
  std::unique_lock<std::mutex> entityLockGuard(entityMutex_);
  auto it = entityMap_.find(gId);
  if (it == entityMap_.end()) {
    throw std::runtime_error("Entity::getEntity Enttiy "+std::to_string(gId)+" does not exist");
  }
  return it->second.get();
}

size_t Entity::size() const {
  return entityMap_.size();
}

const std::map<Entity::EntityId, std::shared_ptr<packet::parsing::Object>>& Entity::getEntityMap() const {
  return entityMap_;
}

} // namespace state