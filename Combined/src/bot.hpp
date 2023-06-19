#ifndef BOT_HPP_
#define BOT_HPP_

#include "broker/packetBroker.hpp"
#include "config/config.hpp"
#include "event/event.hpp"
#include "packet/building/commonBuilding.hpp"
#include "packetProcessor.hpp"
#include "pk2/gameData.hpp"
#include "pk2/gameData.hpp"
#include "proxy.hpp"
#include "statAggregator.hpp"
#include "state/machine/autoPotion.hpp"
#include "state/machine/botting.hpp"
#include "state/worldState.hpp"

#include <optional>
#include <vector>

class Bot {
public:
  Bot(const config::Config &config,
      const pk2::GameData &gameData,
      Proxy &proxy,
      broker::PacketBroker &packetBroker,
      broker::EventBroker &eventBroker);

  void initialize();
  const config::Config& config() const;
  const proto::config::CharacterConfig& currentCharacterConfig() const;
  const pk2::GameData& gameData() const;
  Proxy& proxy() const;
  broker::PacketBroker& packetBroker() const;
  broker::EventBroker& eventBroker();
  const state::WorldState& worldState() const;
  state::EntityTracker& entityTracker();
  const state::EntityTracker& entityTracker() const;
  state::Self& selfState();
  const state::Self& selfState() const;
protected:
  friend class broker::EventBroker;
  void handleEvent(const event::Event *event);

  const config::Config &config_;
  const pk2::GameData &gameData_;
  Proxy &proxy_;
  broker::PacketBroker &packetBroker_;
  broker::EventBroker &eventBroker_;
  state::WorldState worldState_{gameData_, eventBroker_}; // TODO: For multi-character, this will move out of the bot
  PacketProcessor packetProcessor_{worldState_, packetBroker_, eventBroker_, gameData_};
  StatAggregator statAggregator_{worldState_, eventBroker_};

private:
  std::unique_ptr<state::machine::StateMachine> autoPotionStateMachine_;
  std::unique_ptr<state::machine::StateMachine> bottingStateMachine_;
  inline static const std::string kEstVisRangeFilename{"estimatedVisibilityRange.txt"};

  void subscribeToEvents();

  // Main logic
  void onUpdate(const event::Event *event = nullptr);

  // Bot actions from UI
  void handleRequestStartTraining();
  void handleRequestStopTraining();
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
  void handleSpawned(const event::Event *event);
  void handleCosSpawned(const event::CosSpawned &event);
  void handleVitalsChanged();
  void handleStatesChanged();

  // Skills
  void handleSkillBegan(const event::SkillBegan &event);
  void handleSkillEnded(const event::SkillEnded &event);
  void handleSkillCooldownEnded(const event::SkillCooldownEnded &event);

  // Misc
  void entitySpawned(const event::EntitySpawned &event);
  void handleBodyStateChanged(const event::EntityBodyStateChanged &event);
  void itemUseTimedOut(const event::ItemUseTimeout &event);
  void handleKnockbackStunEnded();
  void handleKnockdownStunEnded();
  void handleItemCooldownEnded(const event::ItemCooldownEnded &event);

public:
  bool needToGoToTown() const;
  bool similarSkillIsAlreadyActive(sro::scalar_types::ReferenceObjectId skillRefId) const;
  bool canCastSkill(sro::scalar_types::ReferenceObjectId skillRefId) const;
  std::vector<packet::building::NetworkReadyPosition> calculatePathToDestination(const sro::Position &destinationPosition) const;
  sro::scalar_types::EntityGlobalId getClosestNpcGlobalId() const;

};

#endif