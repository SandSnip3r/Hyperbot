#ifndef UI_USERINTERFACE_HPP_
#define UI_USERINTERFACE_HPP_

#include "entity/entity.hpp"
#include "broker/eventBroker.hpp"
#include "packet/structures/packetInnerStructures.hpp"
#include "storage/storage.hpp"

#include "ui-proto/broadcast.pb.h"
#include <silkroad_lib/position.h>
#include <silkroad_lib/scalar_types.h>
#include <zmq.hpp>

#include <optional>
#include <string_view>

namespace pk2 {
class GameData;
} // namespace pk2

namespace state {
class WorldState;
} // namespace state

namespace ui {

class UserInterface {
// TODO: std::string vs std::string_view. Might need to compile protobuf with c++17?
public:
  UserInterface(const pk2::GameData &gameData, broker::EventBroker &eventBroker);
  ~UserInterface();
  void initialize();
  void setWorldState(const state::WorldState &worldState);
  void runAsync();

  void broadcastLaunch();
private:
  zmq::context_t context_;
  zmq::socket_t publisher_{context_, zmq::socket_type::pub};
  const pk2::GameData &gameData_; // TODO: Remove. The actual UserInterface needs this anyways
  broker::EventBroker &eventBroker_;
  const state::WorldState *worldState_;
  std::thread thr_;

  void subscribeToEvents();
  void handleEvent(const event::Event *event);

  void run();
  void handleRequest(const zmq::message_t &request);

  void handleSelfSpawned();
  void handleCosSpawned(const event::CosSpawned &event);
  void handleEntitySpawned(const event::EntitySpawned &event);
  void handleEntityMovementBegan(const event::EntityMovementBegan &event);
  void handleEntityMovementEnded(const event::EntityMovementEnded &event);
  void handleEntityPositionUpdated(sro::scalar_types::EntityGlobalId globalId);
  void handleEntityNotMovingAngleChanged(sro::scalar_types::EntityGlobalId globalId);
  void handleStorageInitialized();
  void handleGuildStorageInitialized();
  void handleInventoryUpdated(const event::InventoryUpdated &inventoryUpdatedEvent);
  void handleAvatarInventoryUpdated(const event::AvatarInventoryUpdated &avatarInventoryUpdatedEvent);
  void handleCosInventoryUpdated(const event::CosInventoryUpdated &cosInventoryUpdatedEvent);
  void handleStorageUpdated(const event::StorageUpdated &storageUpdatedEvent);
  void handleGuildStorageUpdated(const event::GuildStorageUpdated &guildStorageUpdatedEvent);
  void handleWalkingPathUpdated(const event::WalkingPathUpdated &walkingPathUpdatedEvent);

  void broadcastItemUpdateForSlot(proto::broadcast::ItemLocation itemLocation, const storage::Storage &itemStorage, const uint8_t slotIndex);
  void broadcastCharacterSpawn();
  void broadcastCharacterHpUpdate(uint32_t currentHp);
  void broadcastCharacterMpUpdate(uint32_t currentMp);
  void broadcastCharacterMaxHpMpUpdate(uint32_t maxHp, uint32_t maxMp);
  void broadcastCharacterLevelUpdate(uint8_t currentLevel);
  void broadcastCharacterExperienceUpdate(uint64_t currentExperience, uint32_t currentSpExperience);
  void broadcastCharacterSpUpdate(uint32_t skillPoints);
  void broadcastCharacterNameUpdate(std::string_view characterName);
  void broadcastGoldAmountUpdate(uint64_t goldAmount, proto::broadcast::ItemLocation goldLocation);
  void broadcastPositionChangedUpdate(const sro::Position &currentPosition);
  void broadcastMovementBeganUpdate(const sro::Position &srcPosition, const sro::Position &destPosition, float speed);
  void broadcastMovementBeganUpdate(const sro::Position &srcPosition, sro::Angle angle, float speed);
  void broadcastMovementEndedUpdate(const sro::Position &currentPosition);
  void broadcastNotMovingAngleChangedUpdate(sro::Angle angle);
  void broadcastRegionNameUpdate(std::string_view regionName);
  void broadcastItemUpdate(proto::broadcast::ItemLocation itemLocation, uint8_t slotIndex, uint16_t quantity, std::optional<std::string> itemName={});
  void broadcastEntitySpawned(const entity::Entity *entity);
  void broadcastEntityDespawned(uint32_t globalId);
  void broadcastEntityPositionChanged(const sro::scalar_types::EntityGlobalId globalId, const sro::Position &position);
  void broadcastEntityMovementBegan(const sro::scalar_types::EntityGlobalId globalId, const sro::Position &srcPosition, const sro::Position &destPosition, float speed);
  void broadcastEntityMovementBegan(const sro::scalar_types::EntityGlobalId globalId, const sro::Position &srcPosition, sro::Angle angle, float speed);
  void broadcastEntityMovementEnded(const sro::scalar_types::EntityGlobalId globalId, const sro::Position &currentPosition);
  void broadcastEntityLifeStateChanged(const sro::scalar_types::EntityGlobalId globalId, const sro::entity::LifeState lifeState);
  void broadcastTrainingAreaSet(const entity::Geometry *trainingAreaGeometry);
  void broadcastTrainingAreaReset();
  void broadcastStateMachineCreated(const std::string &stateMachineName);
  void broadcastStateMachineDestroyed();
  void broadcastWalkingPathUpdated(const std::vector<sro::Position> &waypoints);

  void broadcast(const proto::broadcast::BroadcastMessage &broadcastProto);

  void setPosition(proto::broadcast::Position *msg, const sro::Position &pos) const;
  void setCharacterMovementBegan(proto::broadcast::CharacterMovementBegan *msg, const sro::Position &srcPosition, const sro::Position &destPosition, const float speed) const;
  void setCharacterMovementBegan(proto::broadcast::CharacterMovementBegan *msg, const sro::Position &srcPosition, const sro::Angle angle, const float speed) const;
  void setCharacterMovementEnded(proto::broadcast::CharacterMovementEnded *msg, const sro::Position &currentPosition) const;
  void setEntity(proto::entity::Entity *msg, const entity::Entity *entity) const;

};

} // namespace ui

#endif // UI_USERINTERFACE_HPP_