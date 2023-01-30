#ifndef BOT_HPP_
#define BOT_HPP_

#include "broker/packetBroker.hpp"
#include "config/characterLoginData.hpp"
#include "event/event.hpp"
#include "packetProcessor.hpp"
#include "pk2/gameData.hpp"
#include "proxy.hpp"
#include "state/machine/autoPotion.hpp"
#include "state/machine/botting.hpp"
#include "state/worldState.hpp"

#include <optional>
class Bot {
public:
  Bot(const config::CharacterLoginData &loginData,
      const pk2::GameData &gameData,
      Proxy &proxy,
      broker::PacketBroker &packetBroker,
      broker::EventBroker &eventBroker);

  void initialize();
  const pk2::GameData& gameData() const;
  Proxy& proxy() const;
  broker::PacketBroker& packetBroker() const;
  broker::EventBroker& eventBroker();
  const state::WorldState& worldState() const;
  state::EntityTracker& entityTracker();
  state::Self& selfState();
protected:
  friend class broker::EventBroker;
  void handleEvent(const event::Event *event);

  const config::CharacterLoginData &loginData_; // TODO: Move this into a configuration object
  const pk2::GameData &gameData_;
  Proxy &proxy_;
  broker::PacketBroker &packetBroker_;
  broker::EventBroker &eventBroker_;
  state::WorldState worldState_{gameData_, eventBroker_}; // TODO: For multi-character, this will move out of the bot
  PacketProcessor packetProcessor_{worldState_, packetBroker_, eventBroker_, gameData_};

private:
  state::machine::AutoPotion autoPotionStateMachine_{*this};
  std::unique_ptr<state::machine::StateMachine> bottingStateMachine_;

  void subscribeToEvents();

  // Main logic
  void onUpdate(const event::Event *event = nullptr);

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
  void handleEntityMovementBegan(const event::EntityMovementBegan &event);
  void handleEntityMovementTimerEnded(sro::scalar_types::EntityGlobalId globalId);
  void handleEntityEnteredGeometry(const event::EntityEnteredGeometry &event);
  void handleEntityExitedGeometry(const event::EntityExitedGeometry &event);
  // Character info events
  void handleSpawned();
  void handleCosSpawned(const event::CosSpawned &event);
  void handleVitalsChanged();
  void handleStatesChanged();

  // Skills
  void handleSkillBegan(const event::SkillBegan &event);
  void handleSkillEnded(const event::SkillEnded &event);
  void handleSkillCooldownEnded(const event::SkillCooldownEnded &event);

  // Misc
  void entitySpawned(const event::EntitySpawned &event);
  void itemUseTimedOut(const event::ItemUseTimeout &event);
  void handleKnockbackStunEnded();
  void handleKnockdownStunEnded();
  void handleItemCooldownEnded(const event::ItemCooldownEnded &event);

};

#endif