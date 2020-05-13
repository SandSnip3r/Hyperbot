#ifndef CHARACTER_INFO_MODULE_HPP_
#define CHARACTER_INFO_MODULE_HPP_

#include "../broker/eventBroker.hpp"
#include "../broker/packetBroker.hpp"
#include "../packet/parsing/packetParser.hpp"
#include "../packet/parsing/parsedPacket.hpp"
#include "../packet/parsing/serverAgentCharacterData.hpp"
#include "../packet/parsing/serverAgentEntityUpdateState.hpp"
#include "../pk2/gameData.hpp"
#include "../shared/silkroad_security.h"
#include "../state/entity.hpp"
#include "../state/self.hpp"
#include "../storage/buybackQueue.hpp"
#include "../storage/item.hpp"
#include "../storage/storage.hpp"

#include <array>
#include <optional>
#include <deque>
#include <mutex>
#include <vector>

namespace module {

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
  CharacterInfoModule(state::Entity &entityState,
                      state::Self &selfState,
                      storage::Storage &inventory,
                      broker::PacketBroker &brokerSystem,
                      broker::EventBroker &eventBroker,
                      const packet::parsing::PacketParser &packetParser,
                      const pk2::GameData &gameData);
private:
  static const int kEuPotionDefaultDelayMs_{15000};
  static const int kChPotionDefaultDelayMs_{1000};
  // TODO: Maybe we should make this member data and try to improve it based on item use results
  static const int kPotionDelayBufferMs_ = 225; //200 too fast sometimes, 300 seems always good

  //******************************************************************************************
  //***************************************Configuration**************************************
  //******************************************************************************************
  // Potion configuration
  const double kHpThreshold_{0.90};
  const double kMpThreshold_{0.80};
  const double kVigorThreshold_{0.40};
  //******************************************************************************************

  state::Entity &entityState_;
  state::Self &selfState_;
  broker::PacketBroker &broker_;
  broker::EventBroker &eventBroker_;
  const packet::parsing::PacketParser &packetParser_;
  const pk2::GameData &gameData_;
  std::mutex contentionProtectionMutex_;

  // Pills
  std::optional<broker::TimerManager::TimerId> universalPillEventId_, purificationPillEventId_;

  // Potions
  std::optional<broker::TimerManager::TimerId> hpPotionEventId_, mpPotionEventId_, vigorPotionEventId_;
  int potionDelayMs_{1000};
  
  // States
  // Bitmask of all states (initialized as having no states)
  // uint32_t stateBitmask_{0};
  // Set all states as effect/level 0 (meaning there is no state)
  // std::array<uint16_t,6> legacyStateEffects_ = {0};
  std::array<uint8_t,32> modernStateLevel_ = {0};

  std::deque<UsedItem> usedItemQueue_;

  // User purchasing tracking
  std::optional<packet::parsing::ItemMovement> userPurchaseRequest_;
  std::vector<std::shared_ptr<packet::parsing::Object>> objectsInRange_;

  // Items
  storage::Storage &inventory_;
  uint64_t gold_, storageGold_, guildStorageGold_;
  // storage::Storage storage_;
  storage::BuybackQueue buybackQueue_;

  // Packet handling functions
  bool handlePacket(const PacketContainer &packet);
  void abnormalInfoReceived(const packet::parsing::ParsedServerAbnormalInfo &packet);
  void clientItemMoveReceived(const packet::parsing::ParsedClientItemMove &packet);
  void serverItemMoveReceived(const packet::parsing::ParsedServerItemMove &packet);
  void serverAgentCharacterDataReceived(const packet::parsing::ParsedServerAgentCharacterData &packet);
  void entityUpdateReceived(const packet::parsing::ParsedServerHpMpUpdate &packet);
  void statUpdateReceived(const packet::parsing::ParsedServerAgentCharacterUpdateStats &packet);
  void serverUseItemReceived(const packet::parsing::ParsedServerUseItem &packet);
  void serverAgentGroupSpawnReceived(const packet::parsing::ParsedServerAgentGroupSpawn &packet);
  void serverAgentSpawnReceived(packet::parsing::ParsedServerAgentSpawn &packet);
  void serverAgentDespawnReceived(packet::parsing::ParsedServerAgentDespawn &packet);

  // Event handling functions
  void handleEvent(const event::Event *event);
  void handlePillCooldownEnded(const event::EventCode eventCode);
  void handlePotionCooldownEnded(const event::EventCode eventCode);
  void handleVitalsChanged();
  void handleStatesChanged();

  bool havePotion(PotionType potionType);
  void printGold();
  void serverAgentEntityUpdateStateReceived(packet::parsing::ServerAgentEntityUpdateState &packet);
  void trackObject(std::shared_ptr<packet::parsing::Object> obj);
  void stopTrackingObject(uint32_t gId);
  void resetInventory();
  void initializeInventory(uint8_t inventorySize, const std::map<uint8_t, std::shared_ptr<storage::Item>> &inventoryItemMap);
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
  void setRaceAndGender(uint32_t refObjId);
  void checkIfNeedToUsePill();
  void checkIfNeedToHeal();
  void useUniversalPill();
  void usePurificationPill();
  void usePotion(PotionType potionType);
  void useItem(uint8_t slotNum, uint16_t typeData);
  void updateStates(uint32_t stateBitmask, const std::vector<uint8_t> &stateLevels);
};

} // namespace module

#endif // CHARACTER_INFO_MODULE_HPP_