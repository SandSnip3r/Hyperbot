#ifndef STATE_ENTITY_TRACKER_HPP_
#define STATE_ENTITY_TRACKER_HPP_

#include "entity/entity.hpp"
#include "packet/parsing/parsedPacket.hpp"

#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace state {
  
class EntityTracker {
public:
  void trackEntity(std::shared_ptr<entity::Entity> entity);
  void stopTrackingEntity(sro::scalar_types::EntityGlobalId globalId);
  bool trackingEntity(sro::scalar_types::EntityGlobalId globalId) const;
  entity::Entity* getEntity(sro::scalar_types::EntityGlobalId globalId);
  const entity::Entity* getEntity(sro::scalar_types::EntityGlobalId globalId) const;

  template<typename EntityType>
  EntityType& getEntity(sro::scalar_types::EntityGlobalId globalId);
  template<typename EntityType>
  const EntityType& getEntity(sro::scalar_types::EntityGlobalId globalId) const;

  size_t size() const;
  const std::map<sro::scalar_types::EntityGlobalId, std::shared_ptr<entity::Entity>>& getEntityMap() const;
private:
  mutable std::mutex entityMutex_;
  std::map<sro::scalar_types::EntityGlobalId, std::shared_ptr<entity::Entity>> entityMap_;
  entity::Entity* privateCheckedGetEntity(sro::scalar_types::EntityGlobalId globalId);
  const entity::Entity* privateCheckedGetEntity(sro::scalar_types::EntityGlobalId globalId) const;
};

template<typename EntityType>
EntityType& EntityTracker::getEntity(sro::scalar_types::EntityGlobalId globalId) {
  auto *entity = getEntity(globalId);
  return dynamic_cast<EntityType&>(*entity);
}

template<typename EntityType>
const EntityType& EntityTracker::getEntity(sro::scalar_types::EntityGlobalId globalId) const {
  const auto *entity = getEntity(globalId);
  return dynamic_cast<const EntityType&>(*entity);
}

} // namespace state

#endif // STATE_ENTITY_TRACKER_HPP_