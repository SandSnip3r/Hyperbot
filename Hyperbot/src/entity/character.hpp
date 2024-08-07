#ifndef ENTITY_CHARACTER_HPP_
#define ENTITY_CHARACTER_HPP_

#include "mobileEntity.hpp"
#include "broker/eventBroker.hpp"

#include <silkroad_lib/entity.h>

#include <chrono>
#include <cstdint>
#include <map>
#include <optional>
#include <set>

namespace entity {

class Character : public MobileEntity {
public:
  sro::entity::LifeState lifeState;

  // A Character's HP may not be known.
  virtual bool currentHpIsKnown() const;
  uint32_t currentHp() const;

  void setLifeState(sro::entity::LifeState newLifeState);
  void setCurrentHp(uint32_t hp);

  // ---- Buffs ----
  struct BuffData {
    sro::scalar_types::ReferenceObjectId skillRefId;
    std::chrono::high_resolution_clock::time_point endTimePoint;
  };
  // Maps TokenId to BuffData
  std::map<uint32_t, BuffData> buffDataMap;
  std::set<sro::scalar_types::ReferenceObjectId> activeBuffs() const;
  bool buffIsActive(sro::scalar_types::ReferenceObjectId skillRefId) const;
  int buffMsRemaining(sro::scalar_types::ReferenceObjectId skillRefId) const;
  void addBuff(sro::scalar_types::ReferenceObjectId skillRefId, uint32_t tokenId, int32_t durationMs);
  void removeBuff(sro::scalar_types::ReferenceObjectId skillRefId, uint32_t tokenId);
  void clearBuffs();
  EntityType entityType() const override { return EntityType::kCharacter; }
protected:

  // Current hp might not be known.
  std::optional<uint32_t> currentHp_;
};

} // namespace entity

#endif // ENTITY_CHARACTER_HPP_