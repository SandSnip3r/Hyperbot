#ifndef STATE_WORLD_STATE_HPP_
#define STATE_WORLD_STATE_HPP_

#include "broker/eventBroker.hpp"
#include "entityTracker.hpp"
#include "pk2/gameData.hpp"

#include <silkroad_lib/scalar_types.h>

#include <mutex>
#include <unordered_map>
#include <vector>

namespace state {

class WorldState {
public:
  WorldState(const pk2::GameData &gameData, broker::EventBroker &eventBroker);
  state::EntityTracker& entityTracker();
  const state::EntityTracker& entityTracker() const;

  void addBuff(sro::scalar_types::EntityGlobalId globalId, sro::scalar_types::ReferenceObjectId skillRefId, uint32_t tokenId, int32_t durationMs);
  void removeBuffs(const std::vector<uint32_t> &tokenIds);

  template<typename EntityType = entity::Entity>
  std::shared_ptr<EntityType> getEntity(sro::scalar_types::EntityGlobalId globalId) const {
    return entityTracker_.getEntity<EntityType>(globalId);
  }

  std::mutex mutex;
private:
  const pk2::GameData &gameData_;
  broker::EventBroker &eventBroker_;
  state::EntityTracker entityTracker_;

  // TODO: When an entity despawns, we need to remove their entries from this map
  struct BuffInfo {
    BuffInfo(sro::scalar_types::EntityGlobalId gId, sro::scalar_types::ReferenceObjectId refId) : globalId(gId), skillRefId(refId) {}
    sro::scalar_types::EntityGlobalId globalId;
    sro::scalar_types::ReferenceObjectId skillRefId;
  };
  std::unordered_map<uint32_t, BuffInfo> buffTokenToEntityAndSkillIdMap_;
};

} // namespace state

#endif // STATE_WORLD_STATE_HPP_