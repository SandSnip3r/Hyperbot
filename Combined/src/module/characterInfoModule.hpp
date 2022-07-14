#ifndef CHARACTER_INFO_MODULE_HPP_
#define CHARACTER_INFO_MODULE_HPP_

#include "../broker/eventBroker.hpp"
#include "../broker/packetBroker.hpp"
#include "../packet/parsing/packetParser.hpp"
#include "../packet/parsing/parsedPacket.hpp"
#include "../packet/parsing/serverAgentCharacterData.hpp"
#include "../packet/parsing/serverAgentEntityUpdateState.hpp"
#include "../packet/parsing/serverAgentEntityUpdateMoveSpeed.hpp"
#include "../pk2/gameData.hpp"
#include "../shared/silkroad_security.h"
#include "../state/entity.hpp"
#include "../state/self.hpp"
#include "../storage/item.hpp"
#include "../ui/userInterface.hpp"

#include <array>
#include <optional>
#include <mutex>
#include <vector>

namespace module {

enum class PotionType {
  kHp,
  kMp,
  kVigor
};

class CharacterInfoModule {
public:
  CharacterInfoModule(state::Entity &entityState,
                      state::Self &selfState,
                      broker::PacketBroker &brokerSystem,
                      broker::EventBroker &eventBroker,
                      ui::UserInterface &userInterface,
                      const packet::parsing::PacketParser &packetParser,
                      const pk2::GameData &gameData);
private:

  // TODO: We should move this to a more global configuration area for general bot mechanics configuration
  //       Maybe we could try to improve this value based on item use results
  static const int kPotionDelayBufferMs_ = 225; //200 too fast sometimes, 300 seems always good

  //******************************************************************************************
  //***************************************Configuration**************************************
  //******************************************************************************************
  // TODO: Move to a real config object
  // Potion configuration
  const double kHpThreshold_{0.90};
  const double kMpThreshold_{0.80};
  const double kVigorThreshold_{0.40};
  //******************************************************************************************

  state::Entity &entityState_;
  state::Self &selfState_;
  broker::PacketBroker &broker_;
  broker::EventBroker &eventBroker_;
  ui::UserInterface &userInterface_;
  const packet::parsing::PacketParser &packetParser_;
  const pk2::GameData &gameData_;
  std::mutex contentionProtectionMutex_;

  // Silly temp thing for dropping gold as requested by the UI
  int64_t goldDropAmount_;
  int goldDropRemaining_{0};

  // Packet handling functions
  bool handlePacket(const PacketContainer &packet);
  void abnormalInfoReceived(const packet::parsing::ParsedServerAbnormalInfo &packet);
  void clientItemMoveReceived(const packet::parsing::ParsedClientItemMove &packet);
  void serverItemMoveReceived(const packet::parsing::ParsedServerItemMove &packet);
  void serverAgentCharacterDataReceived(const packet::parsing::ParsedServerAgentCharacterData &packet);
  void entityUpdateReceived(const packet::parsing::ParsedServerHpMpUpdate &packet);
  void statUpdateReceived(const packet::parsing::ParsedServerAgentCharacterUpdateStats &packet);
  void serverAgentEntityUpdateMoveSpeedReceived(const packet::parsing::ServerAgentEntityUpdateMoveSpeed &packet);
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
  void handleDropGold(const event::Event *event);

  bool havePotion(PotionType potionType);
  void printGold();
  void serverAgentEntityUpdateStateReceived(packet::parsing::ServerAgentEntityUpdateState &packet);
  void trackObject(std::shared_ptr<packet::parsing::Object> obj);
  void stopTrackingObject(uint32_t gId);
  void initializeInventory(uint8_t inventorySize, const std::map<uint8_t, std::shared_ptr<storage::Item>> &inventoryItemMap);
  int getGrainDelay();
  int getUniversalPillDelay();
  int getPurificationPillDelay();
  bool alreadyUsedUniversalPill();
  bool alreadyUsedPurificationPill();
  bool alreadyUsedPotion(PotionType potionType);
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