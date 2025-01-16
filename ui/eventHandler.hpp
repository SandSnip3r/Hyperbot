#ifndef EVENT_HANDLER_HPP_
#define EVENT_HANDLER_HPP_

#include "ui_proto/broadcast.pb.h"
// #include "ui_proto/old_config.pb.h"

#include <silkroad_lib/entity.h>
#include <silkroad_lib/position.h>
#include <silkroad_lib/scalar_types.h>

#include <zmq.hpp>

#include <QObject>

#include <mutex>
#include <optional>
#include <string>
#include <thread>

class EventHandler : public QObject {
  Q_OBJECT
public:
  EventHandler(zmq::context_t &context);
  ~EventHandler();

  void runAsync();
signals:
  void connected();
  void launch();
  void characterSpawn();
  void characterHpUpdateChanged(uint32_t currentHp);
  void characterMpUpdateChanged(uint32_t currentMp);
  void characterMaxHpMpUpdateChanged(uint32_t maxHp, uint32_t maxMp);
  void characterLevelUpdate(int32_t level, int64_t expRequired);
  void characterExperienceUpdate(uint64_t currentExperience, uint32_t currentSpExperience);
  void characterSpUpdate(uint32_t skillPoints);
  void characterNameUpdate(const std::string &name);
  void inventoryGoldAmountUpdate(uint64_t goldAmount);
  void storageGoldAmountUpdate(uint64_t goldAmount);
  void guildStorageGoldAmountUpdate(uint64_t goldAmount);
  void characterPositionChanged(sro::Position currentPosition);
  void characterMovementBeganToDest(sro::Position currentPosition, sro::Position destinationPosition, float speed);
  void characterMovementBeganTowardAngle(sro::Position currentPosition, uint16_t movementAngle, float speed);
  void characterMovementEnded(sro::Position position);
  void characterNotMovingAngleChanged(sro::Angle angle);
  void regionNameUpdate(const std::string &regionName);
  void characterInventoryItemUpdate(uint8_t slotIndex, uint16_t quantity, std::optional<std::string> itemName);
  void avatarInventoryItemUpdate(uint8_t slotIndex, uint16_t quantity, std::optional<std::string> itemName);
  void cosInventoryItemUpdate(uint8_t slotIndex, uint16_t quantity, std::optional<std::string> itemName);
  void storageItemUpdate(uint8_t slotIndex, uint16_t quantity, std::optional<std::string> itemName);
  void guildStorageItemUpdate(uint8_t slotIndex, uint16_t quantity, std::optional<std::string> itemName);
  void entitySpawned(uint32_t globalId, sro::Position position, proto::entity::Entity entityData);
  void entityDespawned(uint32_t globalId);
  void entityPositionChanged(sro::scalar_types::EntityGlobalId globalId, sro::Position position);
  void entityMovementBeganToDest(sro::scalar_types::EntityGlobalId globalId, sro::Position currentPosition, sro::Position destinationPosition, float speed);
  void entityMovementBeganTowardAngle(sro::scalar_types::EntityGlobalId globalId, sro::Position currentPosition, uint16_t movementAngle, float speed);
  void entityMovementEnded(sro::scalar_types::EntityGlobalId globalId, sro::Position position);
  void entityLifeStateChanged(sro::scalar_types::EntityGlobalId globalId, sro::entity::LifeState lifeState);
  void trainingAreaCircleSet(sro::Position center, float radius);
  void trainingAreaReset();
  void stateMachineCreated(std::string name);
  void stateMachineDestroyed();
  void walkingPathUpdated(std::vector<sro::Position> waypoints);
  // void configReceived(proto::old_config::Config config);
private:
  zmq::context_t &context_;
  std::atomic<bool> run_;
  std::thread thr_;
  void run();
  void handle(const proto::broadcast::BroadcastMessage &message);
};

#endif // EVENT_HANDLER_HPP_