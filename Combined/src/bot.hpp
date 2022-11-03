#ifndef BOT_HPP_
#define BOT_HPP_

#include "broker/eventBroker.hpp"
#include "broker/packetBroker.hpp"
#include "config/characterLoginData.hpp"
#include "packetProcessor.hpp"
#include "pk2/gameData.hpp"
#include "proxy.hpp"
#include "state/machine/botting.hpp"
#include "state/worldState.hpp"
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

  const pk2::GameData& gameData() const;
  Proxy& proxy() const;
  broker::PacketBroker& packetBroker() const;
  broker::EventBroker& eventBroker();
  state::EntityTracker& entityTracker();
  state::Self& selfState();
protected:
  friend class broker::EventBroker;
  void handleEvent(const event::Event *event);

  const config::CharacterLoginData &loginData_; // TODO: Move this into a configuration object
  const pk2::GameData &gameData_;
  Proxy &proxy_;
  broker::PacketBroker &packetBroker_;
  broker::EventBroker eventBroker_;
  state::WorldState worldState_{gameData_, eventBroker_}; // TODO: For multi-character, this will move out of the bot
  ui::UserInterface userInterface_{eventBroker_};
  PacketProcessor packetProcessor_{worldState_, packetBroker_, eventBroker_, gameData_};

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
  state::machine::Botting stateMachine_{*this};

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
  void handleLoggedIn();
  void handleStateCharacterListUpdated() const;
  // Movement events
  void handleMovementTimerEnded();
  void handleSpeedUpdated();
  void handleMovementBegan();
  void handleMovementEnded();
  void handleEntityMovementBegan(sro::scalar_types::EntityGlobalId globalId);
  void handleEntityMovementEnded(sro::scalar_types::EntityGlobalId globalId);
  void handleEntityMovementTimerEnded(sro::scalar_types::EntityGlobalId globalId);
  void handleEntityPositionUpdated(sro::scalar_types::EntityGlobalId globalId);
  void handleEntityNotMovingAngleChanged(sro::scalar_types::EntityGlobalId globalId);
  void handleEntityEnteredGeometry(const event::EntityEnteredGeometry &event);
  void handleEntityExitedGeometry(const event::EntityExitedGeometry &event);
  // Character info events
  void handleSpawned();
  void handleCosSpawned(const event::CosSpawned &event);
  void handleItemWaitForReuseDelay(const event::ItemWaitForReuseDelay &event);
  void handlePotionCooldownEnded(const event::EventCode eventCode);
  void handlePillCooldownEnded(const event::EventCode eventCode);
  void handleVitalsChanged();
  void handleStatesChanged();

  // Skills
  void handleSkillBegan(const event::SkillBegan &event);
  void handleSkillEnded(const event::SkillEnded &event);
  void handleSkillCooldownEnded(const event::SkillCooldownEnded &event);

  // Actual action logic
  void checkIfNeedToHeal();
  bool alreadyUsedPotion(PotionType potionType);
  void usePotion(PotionType potionType);

  void checkIfNeedToUsePill();
  bool alreadyUsedUniversalPill();
  bool alreadyUsedPurificationPill();
  void useUniversalPill();
  void usePurificationPill();

  void useItem(uint8_t slotNum, type_id::TypeId typeData);

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
  void entityLifeStateChanged(const event::EntityLifeStateChanged &event);
  void itemUseTimedOut(const event::ItemUseTimeout &event);

};

#endif