#ifndef CHARACTER_INFO_MODULE_HPP_
#define CHARACTER_INFO_MODULE_HPP_

#include "brokerSystem.hpp"
#include "eventBroker.hpp"
#include "gameData.hpp"
#include "packetParser.hpp"
#include "parsedPacket.hpp"
#include "shared/silkroad_security.h"

#include <array>
#include <optional>
#include <deque>
#include <mutex>
#include <vector>

//TODO: Wrap in module namespace

enum class PotionType {
  kHp,
  kMp,
  kVigor
};

enum class Race {
  kChinese,
  kEuropean
};

enum class Gender {
  kMale,
  kFemale
};

struct UsedItem {
  UsedItem(uint8_t s, uint16_t i) : slotNum(s), itemData(i) {}
  uint8_t slotNum;
  uint16_t itemData;
};

class CharacterInfoModule {
public:
  CharacterInfoModule(BrokerSystem &brokerSystem,
                      event::EventBroker &eventBroker,
                      const packet::parsing::PacketParser &packetParser,
                      const pk2::media::GameData &gameData);
  bool handlePacket(const PacketContainer &packet);
private:
  static const int kEuPotionDefaultDelayMs_{15000};
  static const int kChPotionDefaultDelayMs_{1000};
  // TODO: Maybe we should make this member data and try to improve it based on item use results
  static const int kPotionDelayBufferMs_ = 225; //200 too fast sometimes, 300 seems always good

  //******************************************************************************************
  //***************************************Configuration**************************************
  //******************************************************************************************
  // Potion configuration
  const double kHpThreshold_{0.80};
  const double kMpThreshold_{0.70};
  const double kVigorThreshold_{0.40};
  //******************************************************************************************

  BrokerSystem &broker_;
  event::EventBroker &eventBroker_;
  const packet::parsing::PacketParser &packetParser_;
  const pk2::media::GameData &gameData_;
  std::mutex contentionProtectionMutex_;

  bool initialized_ = false;
  
  // Character info
  uint32_t uniqueId_{0};
  Race race_;
  Gender gender_;

  // Pills
  std::optional<event::TimerManager::TimerId> universalPillEventId_, purificationPillEventId_;

  // Potions
  std::optional<event::TimerManager::TimerId> hpPotionEventId_, mpPotionEventId_, vigorPotionEventId_;
  int potionDelayMs_{1000};
  
  // Health
  uint32_t hp_{0};
  uint32_t mp_{0};
  uint32_t maxHp_{1};
  uint32_t maxMp_{1};
  
  // States
  // Bitmask of all states (initialized as having no states)
  uint32_t stateBitmask_{0};
  // Set all states as effect/level 0 (meaning there is no state)
  std::array<uint16_t,6> legacyStateEffects_ = {0};
  std::array<uint8_t,32> modernStateLevel_ = {0};

  std::deque<UsedItem> usedItemQueue_;

  std::map<uint8_t, item::Item*> inventoryItemMap_;
  int getHpPotionDelay();
  int getMpPotionDelay();
  int getVigorPotionDelay();
  int getGrainDelay();
  int getUniversalPillDelay();
  int getPurificationPillDelay();
  void updateRace(Race race);
  bool alreadyUsedUniversalPill();
  bool alreadyUsedPurificationPill();
  bool alreadyUsedPotion(PotionType potionType);
  void abnormalInfoReceived(const packet::parsing::ParsedServerAbnormalInfo &packet);
  void handlePillCooldownEnded(const std::unique_ptr<event::Event> &event);
  void handlePotionCooldownEnded(const std::unique_ptr<event::Event> &event);
  void characterInfoReceived(const packet::parsing::ParsedServerAgentCharacterData &packet);
  void entityUpdateReceived(const packet::parsing::ParsedServerHpMpUpdate &packet);
  void statUpdateReceived(const packet::parsing::ParsedServerAgentCharacterUpdateStats &packet);
  void setRaceAndGender(uint32_t refObjId);
  void serverUseItemReceived(const packet::parsing::ParsedServerUseItem &packet);
  void checkIfNeedToUsePill();
  void checkIfNeedToHeal();
  void useUniversalPill();
  void usePurificationPill();
  void usePotion(PotionType potionType);
  void useItem(uint8_t slotNum, uint16_t typeData);
  void updateStates(uint32_t stateBitmask, const std::vector<uint8_t> &stateLevels);
};

#endif // CHARACTER_INFO_MODULE_HPP_