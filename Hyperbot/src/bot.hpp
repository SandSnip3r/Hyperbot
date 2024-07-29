#ifndef BOT_HPP_
#define BOT_HPP_

#include "broker/packetBroker.hpp"
#include "config/characterConfig.hpp"
#include "event/event.hpp"
#include "packet/building/commonBuilding.hpp"
#include "packetProcessor.hpp"
#include "pk2/gameData.hpp"
#include "pk2/gameData.hpp"
#include "proxy.hpp"
#include "sessionId.hpp"
#include "statAggregator.hpp"
#include "state/worldState.hpp"
#include "state/machine/concurrentStateMachines.hpp"
#include "state/machine/stateMachine.hpp"

#include <optional>
#include <string_view>
#include <vector>

// TODO: Principal question: When a new config is received, reinitialize the entire StateMachine tree
//    vs have each state machine constantly pulling realtime values from the config.
class Bot {
public:
  Bot(SessionId sessionId,
      const pk2::GameData &gameData,
      Proxy &proxy,
      broker::PacketBroker &packetBroker,
      broker::EventBroker &eventBroker);

  void initialize();
  void setCharacterToLogin(std::string_view characterName);
  const config::CharacterConfig* config() const;
  const pk2::GameData& gameData() const;
  Proxy& proxy() const;
  broker::PacketBroker& packetBroker() const;
  broker::EventBroker& eventBroker();
  const state::WorldState& worldState() const;
  state::EntityTracker& entityTracker();
  const state::EntityTracker& entityTracker() const;
  state::Self& selfState();
  const state::Self& selfState() const;
  SessionId sessionId() const { return sessionId_; }
protected:
  friend class broker::EventBroker;
  void handleEvent(const event::Event *event);

  std::optional<config::CharacterConfig> config_;
  const SessionId sessionId_;
  const pk2::GameData &gameData_;
  Proxy &proxy_;
  broker::PacketBroker &packetBroker_;
  broker::EventBroker &eventBroker_;
  state::WorldState worldState_{gameData_, eventBroker_}; // TODO: For multi-character, this will move out of the bot
  PacketProcessor packetProcessor_{sessionId_, worldState_, packetBroker_, eventBroker_, gameData_};
  StatAggregator statAggregator_{worldState_, eventBroker_};

private:
  std::unique_ptr<state::machine::StateMachine> loginStateMachine_;
  std::unique_ptr<state::machine::StateMachine> autoPotionStateMachine_;
  std::unique_ptr<state::machine::StateMachine> bottingStateMachine_;
  state::machine::ConcurrentStateMachines concurrentStateMachines_{*this};
  inline static const std::string kEstVisRangeFilename{"estimatedVisibilityRange.txt"};

  void loadConfig(std::string_view characterName);
  void subscribeToEvents();

  // Main logic
  void onUpdate(const event::Event *event = nullptr);

  // Bot actions from UI
  void handleRequestStartTraining();
  void handleRequestStopTraining();
  void startTraining();
  void stopTraining();

  // Chat commands
  void handleChatCommand(const event::ChatReceived &event);

  // Debug help
  void handleInjectPacket(const event::InjectPacket &castedEvent);
  // Login events
  void handleLoggedIn(const event::Event *event);
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
  void handleSkillEnded(const event::SkillEnded &event);
  void handleSkillCooldownEnded(const event::SkillCooldownEnded &event);

  // Misc
  void entitySpawned(const event::EntitySpawned &event);
  void handleBodyStateChanged(const event::EntityBodyStateChanged &event);
  void itemUseTimedOut(const event::ItemUseTimeout &event);
  void handleKnockbackStunEnded();
  void handleKnockdownStunEnded();
  void handleItemCooldownEnded(const event::ItemCooldownEnded &event);
  void handleGameReset(const event::Event *event);
  void setCurrentPositionAsTrainingCenter();
  void handleLearnedSkill(const event::LearnSkillSuccess &event);

public:
  bool needToGoToTown() const;
  bool similarSkillIsAlreadyActive(sro::scalar_types::ReferenceObjectId skillRefId) const;
  bool canCastSkill(sro::scalar_types::ReferenceObjectId skillRefId) const;
  std::vector<packet::building::NetworkReadyPosition> calculatePathToDestination(const sro::Position &destinationPosition) const;
  sro::scalar_types::EntityGlobalId getClosestNpcGlobalId() const;

};

#endif