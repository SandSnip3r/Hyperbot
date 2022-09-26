#ifndef BOT_HPP_
#define BOT_HPP_

#include "broker/eventBroker.hpp"
#include "broker/packetBroker.hpp"
#include "config/characterLoginData.hpp"
#include "packet/parsing/packetParser.hpp"
#include "packetProcessor.hpp"
#include "pk2/gameData.hpp"
#include "proxy.hpp"
#include "state/entityTracker.hpp"
#include "state/machine/stateMachine.hpp"
#include "state/self.hpp"
#include "ui/userInterface.hpp"

#include <optional>

// TODO: Move to a better file
enum class PotionType {
  kHp,
  kMp,
  kVigor
};

class Bot {
public:
  Bot(const config::CharacterLoginData &loginData,
      const pk2::GameData &gameData,
      Proxy &proxy,
      broker::PacketBroker &broker);

  state::Self& selfState();
  Proxy& proxy() const;
  broker::PacketBroker& packetBroker() const;
protected:
  friend class broker::EventBroker;
  void handleEvent(const event::Event *event);

  friend class state::machine::CommonStateMachine;
  friend class state::machine::Walking;
  friend class state::machine::TalkingToStorageNpc;
  friend class state::machine::TalkingToShopNpc;
  friend class state::machine::Townlooping;
  const config::CharacterLoginData &loginData_; // TODO: Move this into a configuration object
  const pk2::GameData &gameData_;
  Proxy &proxy_;
  broker::PacketBroker &broker_;
  broker::EventBroker eventBroker_;
  state::EntityTracker entityTracker_;
  state::Self selfState_{eventBroker_, gameData_};
  ui::UserInterface userInterface_{eventBroker_};
  packet::parsing::PacketParser packetParser_{gameData_};
  PacketProcessor packetProcessor_{entityTracker_, selfState_, broker_, eventBroker_, userInterface_, packetParser_, gameData_};

private:
  //******************************************************************************************
  //***************************************Configuration**************************************
  //******************************************************************************************
  // TODO: Move to a real configuration object
  // Potion configuration
  const double kHpThreshold_{0.90};
  const double kMpThreshold_{0.80};
  const double kVigorThreshold_{0.40};
  //******************************************************************************************
  std::optional<state::machine::Townlooping> stateMachine_;

  void subscribeToEvents();

  // Main logic
  void onUpdate(const event::Event *event = nullptr);
  void handleVitals();

  // Bot actions from UI
  void handleStartTraining();
  void handleStopTraining();
  void startTraining();
  void stopTraining();
  // Debug help
  void handleInjectPacket(const event::InjectPacket &castedEvent);
  // Login events
  void handleStateShardIdUpdated() const;
  void handleStateConnectedToAgentServerUpdated();
  void handleStateReceivedCaptchaPromptUpdated() const;
  void handleStateCharacterListUpdated() const;
  // Movement events
  void handleMovementTimerEnded();
  void handleSpeedUpdated();
  void handleMovementBegan();
  void handleMovementEnded();
  void handleEntityMovementBegan(sro::scalar_types::EntityGlobalId globalId);
  void handleEntityMovementEnded(sro::scalar_types::EntityGlobalId globalId);
  void handleEntityMovementTimerEnded(sro::scalar_types::EntityGlobalId globalId);
  void handleEntitySyncedPosition(sro::scalar_types::EntityGlobalId globalId);
  // Character info events
  void handleSpawned();
  void handleCosSpawned(const event::CosSpawned &event);
  void handleItemWaitForReuseDelay(const event::ItemWaitForReuseDelay &event);
  void handlePotionCooldownEnded(const event::EventCode eventCode);
  void handlePillCooldownEnded(const event::EventCode eventCode);
  void handleVitalsChanged();
  void handleStatesChanged();

  // Misc

  // Actual action logic
  void checkIfNeedToHeal();
  bool alreadyUsedPotion(PotionType potionType);
  void usePotion(PotionType potionType);

  void checkIfNeedToUsePill();
  bool alreadyUsedUniversalPill();
  bool alreadyUsedPurificationPill();
  void useUniversalPill();
  void usePurificationPill();

  void useItem(uint8_t slotNum, uint16_t typeData);

  void storageInitialized();
  void guildStorageInitialized();
  void inventoryUpdated(const event::InventoryUpdated &inventoryUpdatedEvent);
  void avatarInventoryUpdated(const event::AvatarInventoryUpdated &avatarInventoryUpdatedEvent);
  void cosInventoryUpdated(const event::CosInventoryUpdated &cosInventoryUpdatedEvent);
  void storageUpdated(const event::StorageUpdated &storageUpdatedEvent);
  void guildStorageUpdated(const event::GuildStorageUpdated &guildStorageUpdatedEvent);
  void broadcastItemUpdateForSlot(broadcast::ItemLocation itemLocation, const storage::Storage &itemStorage, const uint8_t slotIndex);
  void entitySpawned(const event::EntitySpawned &event);
  void entityDespawned(const event::EntityDespawned &event);

};

#endif