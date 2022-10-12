#ifndef STATE_SELF_HPP
#define STATE_SELF_HPP

#include "broker/eventBroker.hpp"
#include "broker/timerManager.hpp"
#include "entity/entity.hpp"
#include "pk2/gameData.hpp"
#include "packet/enums/packetEnums.hpp"
#include "packet/parsing/parsedPacket.hpp"
#include "packet/structures/packetInnerStructures.hpp"
#include "storage/buybackQueue.hpp"
#include "storage/storage.hpp"

#include <silkroad_lib/position.h>
#include <silkroad_lib/scalar_types.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace state {

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
class Self : public entity::PlayerCharacter {
public:
  Self(broker::EventBroker &eventBroker, const pk2::GameData &gameData);
  virtual ~Self() = default;
  void initialize(sro::scalar_types::EntityGlobalId globalId, sro::scalar_types::ReferenceObjectId refObjId);
                  
  // Setters
  void setCurrentLevel(uint8_t currentLevel);
  void setSkillPoints(uint64_t skillPoints);
  void setCurrentExpAndSpExp(uint32_t currentExperience, uint32_t currentSpExperience);

  void resetHpPotionEventId();
  void resetMpPotionEventId();
  void resetVigorPotionEventId();
  void resetUniversalPillEventId();
  void resetPurificationPillEventId();
  void setHpPotionEventId(const broker::TimerManager::TimerId &timerId);
  void setMpPotionEventId(const broker::TimerManager::TimerId &timerId);
  void setVigorPotionEventId(const broker::TimerManager::TimerId &timerId);
  void setUniversalPillEventId(const broker::TimerManager::TimerId &timerId);
  void setPurificationPillEventId(const broker::TimerManager::TimerId &timerId);

  void setHwanSpeed(float hwanSpeed);
  void setBodyState(packet::enums::BodyState bodyState);

  void setMovingToDestination(const std::optional<sro::Position> &sourcePosition, const sro::Position &destinationPosition, broker::EventBroker &eventBroker) override;
  void setMovingTowardAngle(const std::optional<sro::Position> &sourcePosition, const sro::Angle angle, broker::EventBroker &eventBroker) override;

  void setHp(uint32_t hp);
  void setMp(uint32_t mp);
  void setMaxHpMp(uint32_t maxHp, uint32_t maxMp);
  void updateStates(uint32_t stateBitmask, const std::vector<uint8_t> &stateLevels);
  void setStateBitmask(uint32_t stateBitmask);
  void setLegacyStateEffect(packet::enums::AbnormalStateFlag flag, uint16_t effect);
  void setModernStateLevel(packet::enums::AbnormalStateFlag flag, uint8_t level);
  void setMasteriesAndSkills(const std::vector<packet::structures::Mastery> &masteries,
                             const std::vector<packet::structures::Skill> &skills);

  void setGold(uint64_t goldAmount);
  void setStorageGold(uint64_t goldAmount);
  void setGuildStorageGold(uint64_t goldAmount);

  // Getters
  entity::EntityType entityType() const override;
  bool spawned() const;
  Race race() const;
  Gender gender() const;

  uint8_t getCurrentLevel() const;
  uint64_t getSkillPoints() const;
  uint32_t getCurrentExperience() const;
  uint32_t getCurrentSpExperience() const;

  bool haveHpPotionEventId() const;
  bool haveMpPotionEventId() const;
  bool haveVigorPotionEventId() const;
  bool haveUniversalPillEventId() const;
  bool havePurificationPillEventId() const;
  broker::TimerManager::TimerId getHpPotionEventId() const;
  broker::TimerManager::TimerId getMpPotionEventId() const;
  broker::TimerManager::TimerId getVigorPotionEventId() const;
  broker::TimerManager::TimerId getUniversalPillEventId() const;
  broker::TimerManager::TimerId getPurificationPillEventId() const;
  int getHpPotionDelay() const;
  int getMpPotionDelay() const;
  int getVigorPotionDelay() const;
  int getGrainDelay() const;
  int getUniversalPillDelay() const;
  int getPurificationPillDelay() const;

  float hwanSpeed() const;
  packet::enums::BodyState bodyState() const;
  
  uint32_t hp() const;
  uint32_t mp() const;
  std::optional<uint32_t> maxHp() const;
  std::optional<uint32_t> maxMp() const;
  
  uint32_t stateBitmask() const;
  std::array<uint16_t,6> legacyStateEffects() const;
  std::array<uint8_t,32> modernStateLevels() const;

  storage::Storage& getCosInventory(uint32_t globalId);

  uint64_t getGold() const;
  uint64_t getStorageGold() const;
  uint64_t getGuildStorageGold() const;

  std::vector<packet::structures::Mastery> masteries() const;
  std::vector<packet::structures::Skill> skills() const;

  // =================Packets-in-flight state=================
  struct UsedItem {
    UsedItem(uint8_t s, uint16_t i) : inventorySlotNum(s), itemTypeId(i) {}
    uint8_t inventorySlotNum;
    uint16_t itemTypeId;
  };
  // Setters
  void popItemFromUsedItemQueueIfNotEmpty();
  void clearUsedItemQueue();
  void pushItemToUsedItemQueue(uint8_t inventorySlotNum, uint16_t itemTypeId);

  void setUserPurchaseRequest(const packet::structures::ItemMovement &itemMovement);
  void resetUserPurchaseRequest();
  // Getters
  bool usedItemQueueIsEmpty() const;
  bool itemIsInUsedItemQueue(uint16_t itemTypeId) const;
  UsedItem getUsedItemQueueFront() const;

  bool haveUserPurchaseRequest() const;
  packet::structures::ItemMovement getUserPurchaseRequest() const;
  // =========================================================

  std::mutex selfMutex;
  // =======================Log in state======================
  uint16_t shardId;
  bool connectedToAgentServer{false};
  bool receivedCaptchaPrompt{false};
  std::vector<packet::structures::CharacterSelection::Character> characterList;
  bool loggingIn{false};
  uint32_t token;

  // TODO: The two things below do not belong here
  // PC knowledge
  const std::array<uint8_t,6> kMacAddress = {0,0,0,0,0,0};
  // Server knowledge
  const std::string kCaptchaAnswer = "";
  // =========================================================

  bool spawned_{false};
  
  // Character info
private:
  uint8_t currentLevel_;
  uint64_t currentExperience_;
  uint32_t currentSpExperience_;
  uint32_t skillPoints_;
public:
  Race race_;
  Gender gender_;

  // Item use cooldowns
  std::optional<broker::TimerManager::TimerId> hpPotionEventId_, mpPotionEventId_, vigorPotionEventId_;
  std::optional<broker::TimerManager::TimerId> universalPillEventId_, purificationPillEventId_;

  // Speeds
  float hwanSpeed_;

  // Character states
  packet::enums::BodyState bodyState_;

  // Movement/position
  std::optional<broker::TimerManager::TimerId> enteredNewRegionEventId_;

  // Health
private:
  uint32_t hp_;
  uint32_t mp_;
  std::optional<uint32_t> maxHp_;
  std::optional<uint32_t> maxMp_;
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
  std::unordered_map<uint32_t, storage::Storage> cosInventoryMap;

private:
  uint64_t gold_;
  uint64_t storageGold_;
  uint64_t guildStorageGold_;
public:

  // ################################################################################
  // Training state
  bool trainingIsActive{false};

  std::optional<uint32_t> selectedEntity;
  std::optional<uint32_t> pendingTalkGid;
  std::optional<std::pair<uint32_t, packet::enums::TalkOption>> talkingGidAndOption;

  bool haveOpenedStorageSinceTeleport{false};
  // ################################################################################

private:
  broker::EventBroker &eventBroker_;
  const pk2::GameData &gameData_;

  void setRaceAndGender();
  void calculateTimeUntilCollisionWithRegionBoundaryAndPublishDelayedEvent(const sro::Position &currentPosition, double dx, double dy, broker::EventBroker &eventBroker);
  void handleEvent(const event::Event *event);
  void enteredRegion();
  void cancelMovement(broker::EventBroker &eventBroker);
  void checkIfWillLeaveRegionAndSetTimer(broker::EventBroker &eventBroker);

  // Known delay for potion cooldown
  int potionDelayMs_;

  // =================Packets-in-flight state=================
  //  It is required that we keep state of what actions are currently pending
  //  TODO: It could be a good idea to separate this out somehow

  // usedItemQueue_ is a list of items that we sent a packet to use but havent yet heard back from the server on their success or failure
  std::deque<UsedItem> usedItemQueue_;

  // User purchasing tracking
  // This is for tracking an item that was most recently purchased from an NPC by the human interacting with the client
  //  Once the BuyFromNpc item movement packet comes in, we match that item movement with this item movement
  std::optional<packet::structures::ItemMovement> userPurchaseRequest_;
  // =========================================================

  // Game knowledge
  // TODO: This is general game knowledge. Consider moving to GameData.
  static const int kEuPotionDefaultDelayMs_{15000};
  static const int kChPotionDefaultDelayMs_{1000};
};

} // namespace state

#endif // STATE_SELF_HPP