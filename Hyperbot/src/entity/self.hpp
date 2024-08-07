#ifndef ENTITY_SELF_HPP_
#define ENTITY_SELF_HPP_

#include "broker/eventBroker.hpp"
#include "broker/timerManager.hpp"
#include "entity/playerCharacter.hpp"
#include "entity/geometry.hpp"
#include "pk2/gameData.hpp"
#include "packet/enums/packetEnums.hpp"
#include "packet/parsing/parsedPacket.hpp"
#include "packet/structures/packetInnerStructures.hpp"
#include "state/skillEngine.hpp"
#include "storage/buybackQueue.hpp"
#include "storage/storage.hpp"

#include <silkroad_lib/position.h>
#include <silkroad_lib/scalar_types.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace entity {

enum class Race {
  kChinese,
  kEuropean
};

enum class Gender {
  kMale,
  kFemale
};

// TODO: It will probably make more sense to lock the character state more broadly
//  For example, when a packet comes in and we're updating the state
//  Or, when we're doing game logic
class Self : public PlayerCharacter {
public:
  Self(const pk2::GameData &gameData, sro::scalar_types::EntityGlobalId globalId, sro::scalar_types::ReferenceObjectId refObjId, uint32_t jId);
  ~Self() override;

  // The initialize functions are meant to be called during construction. No events will be published during these.
  void initializeCurrentHp(uint32_t hp);
  void initializeCurrentMp(uint32_t mp);
  void initializeCurrentLevel(uint8_t currentLevel);
  void initializeSkillPoints(uint32_t skillPoints);
  void initializeAvailableStatPoints(uint16_t statPoints);
  void initializeHwanPoints(uint8_t hwanPoints);
  void initializeCurrentExpAndSpExp(uint64_t currentExperience, uint64_t currentSpExperience);
  void initializeBodyState(packet::enums::BodyState bodyState);
  void initializeGold(uint64_t goldAmount);

  void initializeEventBroker(broker::EventBroker &eventBroker) override;
                  
  // Setters
  void setCurrentLevel(uint8_t currentLevel);
  void setHwanLevel(uint8_t hwanLevel);
  void setSkillPoints(uint32_t skillPoints);
  void setAvailableStatPoints(uint16_t statPoints);
  void setCurrentExpAndSpExp(uint64_t currentExperience, uint64_t currentSpExperience);

  void setHwanSpeed(float hwanSpeed);
  void setBodyState(packet::enums::BodyState bodyState);
  void setHwanPoints(uint8_t hwanPoints);

  void setMovingToDestination(const std::optional<sro::Position> &sourcePosition, const sro::Position &destinationPosition) override;
  void setMovingTowardAngle(const std::optional<sro::Position> &sourcePosition, const sro::Angle angle) override;

  void setCurrentMp(uint32_t mp);
  void setMaxHpMp(uint32_t maxHp, uint32_t maxMp);
  void setStatPoints(uint16_t strPoints, uint16_t intPoints);
  void updateStates(uint32_t stateBitmask, const std::vector<uint8_t> &stateLevels);
  void setStateBitmask(uint32_t stateBitmask);
  void setLegacyStateEffect(packet::enums::AbnormalStateFlag flag, uint16_t effect);
  void setModernStateLevel(packet::enums::AbnormalStateFlag flag, uint8_t level);
  void setMasteriesAndSkills(const std::vector<packet::structures::Mastery> &masteries,
                             const std::vector<packet::structures::Skill> &skills);
  void learnSkill(sro::scalar_types::ReferenceSkillId skillId);
  void learnMastery(sro::scalar_types::ReferenceMasteryId masteryId, uint8_t masteryLevel);

  void setGold(uint64_t goldAmount);
  void setStorageGold(uint64_t goldAmount);
  void setGuildStorageGold(uint64_t goldAmount);

  void usedAnItem(type_id::TypeId typeData, std::optional<std::chrono::milliseconds> cooldown);
  void itemCooldownEnded(type_id::TypeId itemTypeData);

  // Getters
  EntityType entityType() const override { return EntityType::kSelf; }
  bool spawned() const;
  Race race() const;
  Gender gender() const;

  uint8_t getCurrentLevel() const;
  uint8_t hwanLevel() const;
  uint32_t getSkillPoints() const;
  uint16_t getAvailableStatPoints() const;
  uint64_t getCurrentExperience() const;
  uint64_t getCurrentSpExperience() const;

  bool haveHpPotionEventId() const;
  bool haveMpPotionEventId() const;
  bool haveVigorPotionEventId() const;
  bool haveUniversalPillEventId() const;
  bool havePurificationPillEventId() const;
  int getHpPotionDelay() const;
  int getMpPotionDelay() const;
  int getVigorPotionDelay() const;
  int getHpGrainDelay() const;
  int getMpGrainDelay() const;
  int getVigorGrainDelay() const;
  int getUniversalPillDelay() const;
  int getPurificationPillDelay() const;

  float hwanSpeed() const;
  packet::enums::BodyState bodyState() const;
  uint8_t hwanPoints() const;
  
  // Self's HP is always known.
  bool currentHpIsKnown() const override { return true; }
  // Self's MP is always known.
  uint32_t currentMp() const;
  std::optional<uint32_t> maxHp() const;
  std::optional<uint32_t> maxMp() const;
  std::optional<uint16_t> strPoints() const;
  std::optional<uint16_t> intPoints() const;
  
  uint32_t stateBitmask() const;
  std::array<uint16_t,6> legacyStateEffects() const;
  std::array<uint8_t,32> modernStateLevels() const;

  storage::Storage& getCosInventory(uint32_t globalId);

  uint64_t getGold() const;
  uint64_t getStorageGold() const;
  uint64_t getGuildStorageGold() const;

  std::vector<packet::structures::Mastery> masteries() const;
  std::vector<packet::structures::Skill> skills() const;
  bool haveSkill(sro::scalar_types::ReferenceObjectId id) const;
  uint8_t getMasteryLevel(sro::scalar_types::ReferenceMasteryId id) const;

  bool canUseItems() const;
  bool canUseItem(type_id::TypeId itemTypeId) const;
  bool canUseItem(type_id::TypeCategory itemType) const;

  // =================Packets-in-flight state=================
  struct UsedItem {
    UsedItem(sro::scalar_types::StorageIndexType s, type_id::TypeId i) : inventorySlotNum(s), typeId(i) {}
    sro::scalar_types::StorageIndexType inventorySlotNum;
    type_id::TypeId typeId;
  };
  // Setters
  void setUserPurchaseRequest(const packet::structures::ItemMovement &itemMovement);
  void resetUserPurchaseRequest();
  // Getters
  bool haveUserPurchaseRequest() const;
  packet::structures::ItemMovement getUserPurchaseRequest() const;
  // =========================================================

  // mutable std::mutex selfMutex;

  bool spawned_{true}; // TODO: Remove
  
  // Character info
private:
  uint8_t currentLevel_;
  uint8_t hwanLevel_;
  uint64_t currentExperience_;
  uint64_t currentSpExperience_; // Some packets use uint32, some use uint64.
  uint32_t skillPoints_;
  uint16_t availableStatPoints_;
public:
  uint32_t jId;
  Race race_;
  Gender gender_;

private:
  // Item use cooldowns
  std::map<type_id::TypeId, broker::EventBroker::EventId> itemCooldownEventIdMap_;
public:

  // Speeds
  float hwanSpeed_;

  // Character states
  packet::enums::BodyState bodyState_;
  uint8_t hwanPoints_;

  // Movement/position
  std::optional<broker::EventBroker::EventId> enteredNewRegionEventId_;

  // Health
private:
  uint32_t currentMp_;
  std::optional<uint32_t> maxHp_;
  std::optional<uint32_t> maxMp_;
  std::optional<uint16_t> strPoints_;
  std::optional<uint16_t> intPoints_;
public:

  // Statuses
  // Bitmask of all states (initialized as having no states)
  uint32_t stateBitmask_{0};
  // Set all states as effect/level 0 (meaning there is no state)
  std::array<uint16_t,6> legacyStateEffects_ = {0};
  std::array<uint8_t,32> modernStateLevels_ = {0};

  // Skills
  std::vector<packet::structures::Mastery> masteries_;
  std::vector<packet::structures::Skill> skills_;

  // Inventory
  storage::Storage inventory, storage, guildStorage;
  storage::Storage avatarInventory;
  storage::BuybackQueue buybackQueue;

  // COS
  // TODO: Create a proper COS object
  // Map from COS global Id to its inventory
  std::unordered_map<sro::scalar_types::EntityGlobalId, storage::Storage> cosInventoryMap;

private:
  uint64_t gold_;
  uint64_t storageGold_;
  uint64_t guildStorageGold_;
public:
  // Training state
  bool trainingIsActive{false};

  std::optional<uint32_t> selectedEntity;
  std::optional<uint32_t> pendingTalkGid;
  std::optional<std::pair<uint32_t, packet::enums::TalkOption>> talkingGidAndOption;

  bool haveOpenedStorageSinceTeleport{false};

  // Training area
  std::unique_ptr<Geometry> trainingAreaGeometry;
  void setTrainingAreaGeometry(std::unique_ptr<Geometry> &&geometry);
  void resetTrainingAreaGeometry();

  // Skills
  // TODO: Skill cooldowns transcend the lifetime of the self entity (or any entity). This should not exist inside the entity.
  state::SkillEngine skillEngine;

  // Misc
  // TODO: Remove. This is a temporary mechanism to measure the maximum visibility range.
  double estimatedVisibilityRange{876.879};
  bool stunnedFromKnockdown{false};
  bool stunnedFromKnockback{false};

  bool inTown() const;

private:
  const pk2::GameData &gameData_;
  std::optional<broker::EventBroker::SubscriptionId> enteredRegionSubscriptionId_;

  void setRaceAndGender();
  void calculateTimeUntilCollisionWithRegionBoundaryAndPublishDelayedEvent(const sro::Position &currentPosition, double dx, double dy);
  void handleEvent(const event::Event *event);
  void enteredRegion(const event::Event *event);
  void cancelEvents();
  void cancelMovement() override;
  void checkIfWillLeaveRegionAndSetTimer();

  // Known delay for potion cooldown
  int potionDelayMs_;

  // =================Packets-in-flight state=================
  //  It is required that we keep state of what actions are currently pending
  //  TODO: It could be a good idea to separate this out somehow

  // User purchasing tracking
  // This is for tracking an item that was most recently purchased from an NPC by the human interacting with the client
  //  Once the BuyFromNpc item movement packet comes in, we match that item movement with this item movement
  std::optional<packet::structures::ItemMovement> userPurchaseRequest_;
  // =========================================================

  // Game knowledge
  // TODO: This is general game knowledge. Consider moving to GameData.
  static const int kEuPotionDefaultDelayMs_{15000};
  static const int kChPotionDefaultDelayMs_{1000};
  static const int kGrainDelayMs_{4000};
  static const int kPanicPotionDelayIncreaseMs_{4000};
  static const int kCombustionPotionDelayIncreaseMs_{4000};
};

} // namespace entity

#endif // ENTITY_SELF_HPP_