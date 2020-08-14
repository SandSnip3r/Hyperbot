#ifndef STATE_SELF_HPP
#define STATE_SELF_HPP

// #include "../packet/parsing/parsedPacket.hpp" // Object
#include "../pk2/gameData.hpp"
#include "../packet/enums/packetEnums.hpp"
#include "../packet/structures/packetInnerStructures.hpp"

#include <array>
#include <cstdint>
#include <mutex>
#include <optional>

namespace state {

enum class Race {
  kChinese,
  kEuropean
};

enum class Gender {
  kMale,
  kFemale
};

int toBitNum(packet::enums::AbnormalStateFlag stateFlag);
packet::enums::AbnormalStateFlag fromBitNum(int n);
  
class Self {
public:
  Self(const pk2::GameData &gameData);
  void initialize(uint32_t globalId, uint32_t refObjId, uint32_t hp, uint32_t mp, const std::vector<packet::structures::Mastery> &masteries, const std::vector<packet::structures::Skill> &skills);
  void setLifeState(packet::enums::LifeState lifeState);
  void setBodyState(packet::enums::BodyState bodyState);
  void setHp(uint32_t hp);
  void setMp(uint32_t mp);
  void setMaxHpMp(uint32_t maxHp, uint32_t maxMp);
  void setStateBitmask(uint32_t stateBitmask);
  void setLegacyStateEffect(packet::enums::AbnormalStateFlag flag, uint16_t effect);
  void setModernStateLevel(packet::enums::AbnormalStateFlag flag, uint8_t level);

  bool spawned() const;
  uint32_t globalId() const;
  Race race() const;
  Gender gender() const;

  packet::enums::LifeState lifeState() const;
  packet::enums::BodyState bodyState() const;
  
  uint32_t hp() const;
  uint32_t mp() const;
  std::optional<uint32_t> maxHp() const;
  std::optional<uint32_t> maxMp() const;
  
  uint32_t stateBitmask() const;
  std::array<uint16_t,6> legacyStateEffects() const;
  std::array<uint8_t,32> modernStateLevels() const;
  
  std::vector<packet::structures::Mastery> masteries() const;
  std::vector<packet::structures::Skill> skills() const;
private:
  const pk2::GameData &gameData_;
  mutable std::mutex selfMutex_;
  bool spawned_{false};
  
  // Character info
  uint32_t globalId_{0};
  Race race_;
  Gender gender_;

  // Character states
  packet::enums::LifeState lifeState_;
  packet::enums::BodyState bodyState_;

  // Health
  uint32_t hp_;
  uint32_t mp_;
  std::optional<uint32_t> maxHp_;
  std::optional<uint32_t> maxMp_;

  // Statuses
  // Bitmask of all states (initialized as having no states)
  uint32_t stateBitmask_{0};
  // Set all states as effect/level 0 (meaning there is no state)
  std::array<uint16_t,6> legacyStateEffects_ = {0};
  std::array<uint8_t,32> modernStateLevels_ = {0};

  // Skills
  std::vector<packet::structures::Mastery> masteries_;
  std::vector<packet::structures::Skill> skills_;

  void setRaceAndGender(uint32_t refObjId);
};

} // namespace state

#endif // STATE_SELF_HPP