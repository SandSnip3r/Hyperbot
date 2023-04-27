#include "logging.hpp"
#include "worldState.hpp"

namespace state {

WorldState::WorldState(const pk2::GameData &gameData, broker::EventBroker &eventBroker) : gameData_(gameData), eventBroker_(eventBroker) {
}

state::EntityTracker& WorldState::entityTracker() {
  return entityTracker_;
}

const state::EntityTracker& WorldState::entityTracker() const {
  return entityTracker_;
}

state::Self& WorldState::selfState() {
  return selfState_;
}

const state::Self& WorldState::selfState() const {
  return selfState_;
}

void WorldState::addBuff(sro::scalar_types::EntityGlobalId globalId, sro::scalar_types::ReferenceObjectId skillRefId, uint32_t tokenId) {
  // For now, we only care about PlayerCharacters
  // TODO: add buffs for others
  entity::Entity *entity = getEntity(globalId);
  if (auto *playerCharacter = dynamic_cast<entity::PlayerCharacter*>(entity)) {
    buffTokenToEntityAndSkillIdMap_.emplace(std::piecewise_construct, std::forward_as_tuple(tokenId), std::forward_as_tuple(globalId, skillRefId));
    playerCharacter->addBuff(skillRefId, eventBroker_);
  }
}

void WorldState::removeBuffs(const std::vector<uint32_t> &tokenIds) {
  for (const auto tokenId : tokenIds) {
    auto buffTokenDataIt = buffTokenToEntityAndSkillIdMap_.find(tokenId);
    if (buffTokenDataIt == buffTokenToEntityAndSkillIdMap_.end()) {
      LOG() << "Asked to remove a buff, but we are not tracking it" << std::endl;
      continue;
    }
    entity::Entity *entity = getEntity(buffTokenDataIt->second.globalId);
    auto *playerCharacter = dynamic_cast<entity::PlayerCharacter*>(entity);
    if (playerCharacter == nullptr) {
      throw std::runtime_error("Only tracking buffs for PlayerCharacters, how did this get here?");
    }
    playerCharacter->removeBuff(buffTokenDataIt->second.skillRefId, eventBroker_);
    buffTokenToEntityAndSkillIdMap_.erase(buffTokenDataIt);
  }
}

entity::Entity* WorldState::getEntity(sro::scalar_types::EntityGlobalId globalId) {
  if (globalId == selfState_.globalId) {
    return &selfState_;
  } else if (entityTracker_.trackingEntity(globalId)) {
    return entityTracker_.getEntity(globalId);
  }
}

} // namespace state