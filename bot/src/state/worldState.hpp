#ifndef STATE_WORLD_STATE_HPP_
#define STATE_WORLD_STATE_HPP_

#include "broker/eventBroker.hpp"
#include "entity/character.hpp"
#include "entity/entity.hpp"
#include "entityTracker.hpp"
#include "packet/parsing/serverGatewayShardListResponse.hpp"
#include "pk2/gameData.hpp"

#include <silkroad_lib/scalar_types.hpp>

#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace state {

class WorldState {
public:
  WorldState(const pk2::GameData &gameData, broker::EventBroker &eventBroker);
  state::EntityTracker& entityTracker();
  const state::EntityTracker& entityTracker() const;

  // Returns true if this is the first time we are seeing this entity.
  bool entitySpawned(std::shared_ptr<entity::Entity> entity, broker::EventBroker &eventBroker);
  // Returns true is this is the last time we are seeing this entity.
  bool entityDespawned(sro::scalar_types::EntityGlobalId globalId, broker::EventBroker &eventBroker);

  void addBuff(sro::scalar_types::EntityGlobalId globalId, sro::scalar_types::ReferenceObjectId skillRefId, sro::scalar_types::BuffTokenType tokenId, entity::Character::BuffData::ClockType::time_point castTime);
  void removeBuffs(const std::vector<sro::scalar_types::BuffTokenType> &tokenIds);

  template<typename EntityType = entity::Entity>
  std::shared_ptr<EntityType> getEntity(sro::scalar_types::EntityGlobalId globalId) const {
    return entityTracker_.getEntity<EntityType>(globalId);
  }

  std::mutex mutex;
  std::optional<packet::parsing::ServerGatewayShardListResponse> shardListResponse_;
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
  std::unordered_map<sro::scalar_types::BuffTokenType, BuffInfo> buffTokenToEntityAndSkillIdMap_;
};

} // namespace state

#endif // STATE_WORLD_STATE_HPP_