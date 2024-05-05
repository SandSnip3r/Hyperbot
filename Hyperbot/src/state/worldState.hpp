#ifndef STATE_WORLD_STATE_HPP_
#define STATE_WORLD_STATE_HPP_

#include "broker/eventBroker.hpp"
#include "entityTracker.hpp"
#include "pk2/gameData.hpp"
#include "self.hpp"

#include <silkroad_lib/scalar_types.h>

#include <unordered_map>
#include <vector>

namespace state {

class WorldState {
public:
  WorldState(const pk2::GameData &gameData, broker::EventBroker &eventBroker);
  state::EntityTracker& entityTracker();
  const state::EntityTracker& entityTracker() const;
  state::Self& selfState();
  const state::Self& selfState() const;

  void addBuff(sro::scalar_types::EntityGlobalId globalId, sro::scalar_types::ReferenceObjectId skillRefId, uint32_t tokenId, int32_t durationMs);
  void removeBuffs(const std::vector<uint32_t> &tokenIds);

  entity::Entity* getEntity(sro::scalar_types::EntityGlobalId globalId);
  const entity::Entity* getEntity(sro::scalar_types::EntityGlobalId globalId) const;

  template<typename EntityType>
  EntityType& getEntity(sro::scalar_types::EntityGlobalId globalId);
  template<typename EntityType>
  const EntityType& getEntity(sro::scalar_types::EntityGlobalId globalId) const;
private:
  const pk2::GameData &gameData_;
  broker::EventBroker &eventBroker_;
  state::EntityTracker entityTracker_;
  state::Self selfState_{eventBroker_, gameData_};

  // TODO: When an entity despawns, we need to remove their entries from this map
  struct BuffInfo {
    BuffInfo(sro::scalar_types::EntityGlobalId gId, sro::scalar_types::ReferenceObjectId refId) : globalId(gId), skillRefId(refId) {}
    sro::scalar_types::EntityGlobalId globalId;
    sro::scalar_types::ReferenceObjectId skillRefId;
  };
  std::unordered_map<uint32_t, BuffInfo> buffTokenToEntityAndSkillIdMap_;
};

template<typename EntityType>
EntityType& WorldState::getEntity(sro::scalar_types::EntityGlobalId globalId) {
  return const_cast<EntityType&>(const_cast<const WorldState&>(*this).getEntity<EntityType>(globalId));
}

template<typename EntityType>
const EntityType& WorldState::getEntity(sro::scalar_types::EntityGlobalId globalId) const {
  if (globalId == selfState_.globalId) {
    if (dynamic_cast<const EntityType*>(&selfState_) == nullptr) {
      throw std::runtime_error("Trying to get self entity as an invalid type");
    }
    return dynamic_cast<const EntityType&>(selfState_);
  } else if (entityTracker_.trackingEntity(globalId)) {
    return entityTracker_.getEntity<EntityType>(globalId);
  } else {
    throw std::runtime_error("Trying to get untracked entity");
  }
}

} // namespace state

#endif // STATE_WORLD_STATE_HPP_