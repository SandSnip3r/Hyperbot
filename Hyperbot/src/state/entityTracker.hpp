#ifndef STATE_ENTITY_TRACKER_HPP_
#define STATE_ENTITY_TRACKER_HPP_

#include "entity/entity.hpp"
#include "packet/parsing/parsedPacket.hpp"

#include <map>
#include <memory>
#include <mutex>
#include <type_traits>
#include <vector>

namespace state {
  
class EntityTracker {
public:
  void trackEntity(std::shared_ptr<entity::Entity> entity);
  void stopTrackingEntity(sro::scalar_types::EntityGlobalId globalId);
  bool trackingEntity(sro::scalar_types::EntityGlobalId globalId) const;
  // std::shared_ptr<entity::Entity> getEntity(sro::scalar_types::EntityGlobalId globalId) const;

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
  const std::map<sro::scalar_types::EntityGlobalId, std::shared_ptr<entity::Entity>>& getEntityMap() const; // TODO: Remove
private:
  mutable std::mutex entityMapMutex_;
  std::map<sro::scalar_types::EntityGlobalId, std::shared_ptr<entity::Entity>> entityMap_; // TODO: absl::flat_hash_map
};

} // namespace state

#endif // STATE_ENTITY_TRACKER_HPP_