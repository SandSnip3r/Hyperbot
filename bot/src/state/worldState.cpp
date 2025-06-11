#include "worldState.hpp"

#include "entity/character.hpp"
#include "entity/playerCharacter.hpp"

#include <absl/log/log.h>
#include <absl/strings/str_format.h>
#include <absl/strings/str_join.h>

namespace state {

WorldState::WorldState(const sro::pk2::GameData &gameData, broker::EventBroker &eventBroker) : gameData_(gameData), eventBroker_(eventBroker) {
}

state::EntityTracker& WorldState::entityTracker() {
  return entityTracker_;
}

const state::EntityTracker& WorldState::entityTracker() const {
  return entityTracker_;
}

bool WorldState::entitySpawned(std::shared_ptr<entity::Entity> entity, broker::EventBroker &eventBroker) {
  bool spawned = entityTracker_.entitySpawned(entity, eventBroker, *this);
  if (!spawned) {
    return false;
  }
  // This is a new entity. Check if it has any active buffs.
  if (const auto *character = dynamic_cast<const entity::Character*>(entity.get()); character != nullptr) {
    for (const auto &tokenDataPair : character->buffDataMap) {
      buffTokenToEntityAndSkillIdMap_.emplace(std::piecewise_construct, std::forward_as_tuple(tokenDataPair.first), std::forward_as_tuple(character->globalId, tokenDataPair.second.skillRefId));
    }
  }
  return true;
}

bool WorldState::entityDespawned(sro::scalar_types::EntityGlobalId globalId, broker::EventBroker &eventBroker) {
  const bool despawned = entityTracker_.entityDespawned(globalId, eventBroker);
  if (!despawned) {
    return false;
  }
  // We despawned this entity, remove any buff tokens we were tracking.
  auto it = buffTokenToEntityAndSkillIdMap_.begin();
  while (it != buffTokenToEntityAndSkillIdMap_.end()) {
    if (it->second.globalId == globalId) {
      it = buffTokenToEntityAndSkillIdMap_.erase(it);
    } else {
      ++it;
    }
  }
  return true;
}

void WorldState::addBuff(sro::scalar_types::EntityGlobalId globalId, sro::scalar_types::ReferenceSkillId skillRefId, sro::scalar_types::BuffTokenType tokenId, entity::Character::BuffData::ClockType::time_point castTime) {
  VLOG(1) << "There are " << buffTokenToEntityAndSkillIdMap_.size() << " buffs tracked in WordState";
  // For now, we only care about PlayerCharacters
  // TODO: add buffs for others
  std::shared_ptr<entity::Entity> entity = getEntity(globalId);
  if (auto *character = dynamic_cast<entity::Character*>(entity.get())) {
    if (dynamic_cast<const entity::PlayerCharacter*>(entity.get()) == nullptr) {
      VLOG(1) << "Handling a buff for a non-player character; it is a " << toString(entity->entityType());
    }
    buffTokenToEntityAndSkillIdMap_.emplace(std::piecewise_construct, std::forward_as_tuple(tokenId), std::forward_as_tuple(globalId, skillRefId));
    character->addBuff(skillRefId, tokenId, castTime);
    VLOG(1) << absl::StreamFormat("Buff %s with token %d added to %s", gameData_.getSkillName(skillRefId), tokenId, entity->toString());
  } else {
    throw std::runtime_error("Got a buff for a non-character.");
  }
  VLOG(1) << absl::StreamFormat("Known tokens: [ %s ]", absl::StrJoin(buffTokenToEntityAndSkillIdMap_, ", ", [](std::string *out, const auto data){
    out->append(std::to_string(data.first));
  }));
}

void WorldState::removeBuffs(const std::vector<sro::scalar_types::BuffTokenType> &tokenIds) {
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
    VLOG(1) << absl::StreamFormat("Removing buff %s with token %d from %s", gameData_.getSkillName(buffTokenDataIt->second.skillRefId), tokenId, entity->toString());
    character->removeBuff(buffTokenDataIt->second.skillRefId, tokenId);
    buffTokenToEntityAndSkillIdMap_.erase(buffTokenDataIt);
  }
}

} // namespace state