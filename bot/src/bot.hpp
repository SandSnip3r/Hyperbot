#ifndef BOT_HPP_
#define BOT_HPP_

#include "broker/packetBroker.hpp"
#include "characterLoginInfo.hpp"
#include "config/characterConfig.hpp"
#include "entity/self.hpp"
#include "event/event.hpp"
#include "packet/building/commonBuilding.hpp"
#include "packetProcessor.hpp"
#include "pk2/gameData.hpp"
#include "pk2/gameData.hpp"
#include "proxy.hpp"
#include "common/sessionId.hpp"
#include "statAggregator.hpp"
#include "state/worldState.hpp"
#include "state/machine/sequentialStateMachines.hpp"
#include "state/machine/stateMachine.hpp"

#include <future>
#include <optional>
#include <memory>
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
      broker::EventBroker &eventBroker,
      state::WorldState &worldState);

  void initialize();
  void setCharacter(const CharacterLoginInfo &characterLoginInfo);
  const config::CharacterConfig* config() const;
  const pk2::GameData& gameData() const;
  Proxy& proxy() const;
  broker::PacketBroker& packetBroker() const;
  broker::EventBroker& eventBroker();
  const state::WorldState& worldState() const;
  state::EntityTracker& entityTracker();
  const state::EntityTracker& entityTracker() const;
  std::shared_ptr<entity::Self> selfState() const;
  const storage::Storage& inventory() const;
  SessionId sessionId() const { return sessionId_; }
protected:
  friend class broker::EventBroker;
  void handleEvent(const event::Event *event);

  const SessionId sessionId_;
  const pk2::GameData &gameData_;
  Proxy &proxy_;
  broker::PacketBroker &packetBroker_;
  broker::EventBroker &eventBroker_;
  state::WorldState &worldState_;
  PacketProcessor packetProcessor_{sessionId_, worldState_, packetBroker_, eventBroker_, gameData_};
  // StatAggregator statAggregator_{worldState_, eventBroker_};

  // We track ourself by a pointer to the self entity. Alternatively, we could use the global ID and look up the entity each time. We do not use the global ID because the entity could be removed from the entity tracker before we receive the despawn event.
  // std::optional<sro::scalar_types::EntityGlobalId> selfGlobalId_;
  std::shared_ptr<entity::Self> selfEntity_;

private:
  CharacterLoginInfo characterLoginInfo_;
  std::unique_ptr<state::machine::StateMachine> pvpManagerStateMachine_;

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
  // Character info events
  void handleSelfSpawned(const event::Event *event);
  void handleEntityDespawned(const event::EntityDespawned &event);

  // Misc
  void handleKnockbackStunEnded();
  void handleKnockdownStunEnded();
  void setCurrentPositionAsTrainingCenter();
  void handleLearnedSkill(const event::LearnSkillSuccess &event);

public:
  bool needToGoToTown() const;
  bool similarSkillIsAlreadyActive(sro::scalar_types::ReferenceObjectId skillRefId) const;
  bool canCastSkill(sro::scalar_types::ReferenceObjectId skillRefId) const;
  std::vector<packet::building::NetworkReadyPosition> calculatePathToDestination(const sro::Position &destinationPosition) const;
  sro::scalar_types::EntityGlobalId getClosestNpcGlobalId() const;

  // Interface for RL training.
  std::future<void> asyncOpenClient();
  bool loggedIn() const;
  void asyncStandbyForPvp();

private:
  // Data for RL training interface.
  std::promise<void> clientOpenPromise_;
};

#endif