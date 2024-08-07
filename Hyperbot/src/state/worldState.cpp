#include "worldState.hpp"

#include "entity/character.hpp"
#include "entity/playerCharacter.hpp"

#include <absl/log/log.h>
#include <absl/strings/str_format.h>
#include <absl/strings/str_join.h>

namespace state {

WorldState::WorldState(const pk2::GameData &gameData, broker::EventBroker &eventBroker) : gameData_(gameData), eventBroker_(eventBroker) {
}

state::EntityTracker& WorldState::entityTracker() {
  return entityTracker_;
}

const state::EntityTracker& WorldState::entityTracker() const {
  return entityTracker_;
}

void WorldState::addBuff(sro::scalar_types::EntityGlobalId globalId, sro::scalar_types::ReferenceObjectId skillRefId, uint32_t tokenId, int32_t durationMs) {
  VLOG(1) << "There are " << buffTokenToEntityAndSkillIdMap_.size() << " buffs tracked in WordState";
  // For now, we only care about PlayerCharacters
  // TODO: add buffs for others
  std::shared_ptr<entity::Entity> entity = getEntity(globalId);
  if (auto *character = dynamic_cast<entity::Character*>(entity.get())) {
    if (!dynamic_cast<entity::PlayerCharacter*>(entity.get())) {
      VLOG(1) << "Handling a buff for a non-player character";
    }
    buffTokenToEntityAndSkillIdMap_.emplace(std::piecewise_construct, std::forward_as_tuple(tokenId), std::forward_as_tuple(globalId, skillRefId));
    character->addBuff(skillRefId, tokenId, durationMs);
    VLOG(1) << "Buff with token " << tokenId << " added";
  } else {
    throw std::runtime_error("Got a buff for a non-character.");
  }
  VLOG(1) << absl::StreamFormat("Known tokens: [ %s ]", absl::StrJoin(buffTokenToEntityAndSkillIdMap_, ", ", [](std::string *out, const auto data){
    out->append(std::to_string(data.first));
  }));
}

void WorldState::removeBuffs(const std::vector<uint32_t> &tokenIds) {
  VLOG(1) << absl::StreamFormat("Known tokens: [ %s ]", absl::StrJoin(buffTokenToEntityAndSkillIdMap_, ", ", [](std::string *out, const auto data){
    out->append(std::to_string(data.first));
  }));
  for (const auto tokenId : tokenIds) {
    auto buffTokenDataIt = buffTokenToEntityAndSkillIdMap_.find(tokenId);
    if (buffTokenDataIt == buffTokenToEntityAndSkillIdMap_.end()) {
      VLOG(1) << "Asked to remove a buff " << tokenId << ", but we are not tracking it";
      continue;
    }
    std::shared_ptr<entity::Entity> entity = getEntity(buffTokenDataIt->second.globalId);
    auto *character = dynamic_cast<entity::Character*>(entity.get());
    if (character == nullptr) {
      throw std::runtime_error("Only tracking buffs for Characters, how did this get here?");
    }
    character->removeBuff(buffTokenDataIt->second.skillRefId, tokenId);
    buffTokenToEntityAndSkillIdMap_.erase(buffTokenDataIt);
  }
}

} // namespace state