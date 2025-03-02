#ifndef ENTITY_CHARACTER_HPP_
#define ENTITY_CHARACTER_HPP_

#include "mobileEntity.hpp"
#include "broker/eventBroker.hpp"

#include <silkroad_lib/entity.hpp>
#include <silkroad_lib/scalar_types.hpp>

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
    using ClockType = std::chrono::high_resolution_clock;
    sro::scalar_types::ReferenceSkillId skillRefId;
    std::optional<ClockType::time_point> castTime;
  };
  // Maps TokenId to BuffData
  std::map<sro::scalar_types::BuffTokenType, BuffData> buffDataMap;
  std::set<sro::scalar_types::ReferenceSkillId> activeBuffs() const;
  bool buffIsActive(sro::scalar_types::ReferenceSkillId skillRefId) const;
  std::optional<BuffData::ClockType::time_point> buffCastTime(sro::scalar_types::ReferenceSkillId skillRefId) const;
  void addBuff(sro::scalar_types::ReferenceSkillId skillRefId, sro::scalar_types::BuffTokenType tokenId, std::optional<BuffData::ClockType::time_point> castTime = std::nullopt);
  void removeBuff(sro::scalar_types::ReferenceSkillId skillRefId, sro::scalar_types::BuffTokenType tokenId);
  void clearBuffs();
  EntityType entityType() const override { return EntityType::kCharacter; }
protected:

  // Current hp might not be known.
  std::optional<uint32_t> currentHp_;
};

} // namespace entity

#endif // ENTITY_CHARACTER_HPP_