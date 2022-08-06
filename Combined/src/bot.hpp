#ifndef BOT_HPP_
#define BOT_HPP_

#include "packetProcessor.hpp"
#include "broker/eventBroker.hpp"
#include "broker/packetBroker.hpp"
#include "config/characterLoginData.hpp"
#include "packet/parsing/packetParser.hpp"
#include "pk2/gameData.hpp"
#include "state/entity.hpp"
#include "state/machine/stateMachine.hpp"
#include "state/self.hpp"
#include "ui/userInterface.hpp"

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
      broker::PacketBroker &broker);

protected:
  friend class broker::EventBroker;
  void handleEvent(const event::Event *event);

  friend class state::machine::Walking;
  friend class state::machine::TalkingToStorageNpc;
  friend class state::machine::TalkingToShopNpc;
  friend class state::machine::Townlooping;
  const config::CharacterLoginData &loginData_; // TODO: Move this into a configuration object
  const pk2::GameData &gameData_;
  state::Entity entityState_;
  state::Self selfState_{gameData_};
  broker::PacketBroker &broker_;
  broker::EventBroker eventBroker_;
  ui::UserInterface userInterface_{eventBroker_};
  packet::parsing::PacketParser packetParser_{gameData_};
  PacketProcessor packetProcessor_{entityState_, selfState_, broker_, eventBroker_, userInterface_, packetParser_, gameData_};

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
  state::machine::Townlooping stateMachine_;

  void subscribeToEvents();

  // Main logic
  void onUpdate(const event::Event *event = nullptr);
  void handleVitals();

  // Bot actions from UI
  void handleStartTraining();
  void handleStopTraining();
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
  void handleMovementEnded();
  // Character info events
  void handleItemWaitForReuseDelay(const event::ItemWaitForReuseDelay &castedEvent);
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
};

#endif