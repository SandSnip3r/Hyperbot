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

void WorldState::addBuff(sro::scalar_types::EntityGlobalId globalId, sro::scalar_types::ReferenceObjectId skillRefId, uint32_t tokenId, int32_t durationMs) {
  HYPERBOT_LOG() << "There are " << buffTokenToEntityAndSkillIdMap_.size() << " buffs tracked in WordState" << std::endl;
  // For now, we only care about PlayerCharacters
  // TODO: add buffs for others
  entity::Entity *entity = getEntity(globalId);
  if (auto *character = dynamic_cast<entity::Character*>(entity)) {
    if (!dynamic_cast<entity::PlayerCharacter*>(entity)) {
      HYPERBOT_LOG() << "Handling a buff for a non-player character" << std::endl;
    }
    buffTokenToEntityAndSkillIdMap_.emplace(std::piecewise_construct, std::forward_as_tuple(tokenId), std::forward_as_tuple(globalId, skillRefId));
    character->addBuff(skillRefId, tokenId, durationMs, eventBroker_);
    HYPERBOT_LOG() << "Buff with token " << tokenId << " added" << std::endl;
  } else {
    throw std::runtime_error("Got a buff for a non-character.");
  }
  HYPERBOT_LOG() << "Known tokens: [";
  for (const auto [i,j] : buffTokenToEntityAndSkillIdMap_) {
    std::cout << i << ", ";
  }
  std::cout << "]" <<  std::endl;
}

void WorldState::removeBuffs(const std::vector<uint32_t> &tokenIds) {
  HYPERBOT_LOG() << "Known tokens: [";
  for (const auto [i,j] : buffTokenToEntityAndSkillIdMap_) {
    std::cout << i << ", ";
  }
  std::cout << "]" <<  std::endl;
  for (const auto tokenId : tokenIds) {
    auto buffTokenDataIt = buffTokenToEntityAndSkillIdMap_.find(tokenId);
    if (buffTokenDataIt == buffTokenToEntityAndSkillIdMap_.end()) {
      HYPERBOT_LOG() << "Asked to remove a buff " << tokenId << ", but we are not tracking it" << std::endl;
      continue;
    }
    entity::Entity *entity = getEntity(buffTokenDataIt->second.globalId);
    auto *character = dynamic_cast<entity::Character*>(entity);
    if (character == nullptr) {
      throw std::runtime_error("Only tracking buffs for Characters, how did this get here?");
    }
    character->removeBuff(buffTokenDataIt->second.skillRefId, tokenId, eventBroker_);
    buffTokenToEntityAndSkillIdMap_.erase(buffTokenDataIt);
  }
}

entity::Entity* WorldState::getEntity(sro::scalar_types::EntityGlobalId globalId) {
  return const_cast<entity::Entity*>(const_cast<const WorldState*>(this)->getEntity(globalId));
}

const entity::Entity* WorldState::getEntity(sro::scalar_types::EntityGlobalId globalId) const {
  if (globalId == selfState_.globalId) {
    return &selfState_;
  } else if (entityTracker_.trackingEntity(globalId)) {
    return entityTracker_.getEntity(globalId);
  }
}

} // namespace state