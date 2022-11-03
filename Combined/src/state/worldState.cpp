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
  // For now, we only care about ourself
  // TODO: add buffs for others
  if (globalId == selfState_.globalId) {
    buffTokenToEntityAndSkillIdMap_.emplace(std::piecewise_construct, std::forward_as_tuple(tokenId), std::forward_as_tuple(globalId, skillRefId));
    selfState_.addBuff(skillRefId, eventBroker_);
  }
}

void WorldState::removeBuffs(const std::vector<uint32_t> &tokenIds) {
  for (const auto tokenId : tokenIds) {
    auto buffTokenDataIt = buffTokenToEntityAndSkillIdMap_.find(tokenId);
    if (buffTokenDataIt == buffTokenToEntityAndSkillIdMap_.end()) {
      LOG() << "Asked to remove a buff, but we are not tracking it" << std::endl;
      continue;
    }
    if (buffTokenDataIt->second.globalId != selfState_.globalId) {
      throw std::runtime_error("Only tracking buffs for ourself, how did this get here?");
    }
    selfState_.removeBuff(buffTokenDataIt->second.skillRefId, eventBroker_);
    buffTokenToEntityAndSkillIdMap_.erase(buffTokenDataIt);
  }
}

} // namespace state