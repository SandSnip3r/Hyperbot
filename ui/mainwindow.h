#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "entityData/entity.hpp"
#include "entityData/self.hpp"
#include "eventHandler.hpp"
#include "itemListWidget.hpp"
#include "map/botCharacterGraphicsItem.hpp"
#include "map/trainingAreaGraphicsItem.hpp"
#include "requester.hpp"

#include <silkroad_lib/navmesh/navmesh.h>
#include <silkroad_lib/navmesh/triangulation/navmeshTriangulation.h>
#include <silkroad_lib/position.h>

#include <zmq.hpp>

#include <QGraphicsScene>
#include <QLabel>
#include <QMainWindow>
#include <QPixmap>
#include <QTimer>

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
  Q_OBJECT
public:
  MainWindow(QWidget *parent = nullptr);
  ~MainWindow();

private:
  static const std::filesystem::path kSilkroadPath_;
  static constexpr int kPositionRedrawDelayMs{16};
  static constexpr bool kShowEntityPaths_{true};
  Ui::MainWindow *ui;
  zmq::context_t context_;
  EventHandler eventHandler_{context_};
  Requester requester_{context_};
  entity_data::Self selfData_;
  QTimer *movementUpdateTimer_{nullptr};
  QGraphicsScene *mapScene_{new QGraphicsScene(this)};
  std::optional<sro::navmesh::Navmesh> navmesh_;
  std::optional<sro::navmesh::triangulation::NavmeshTriangulation> navmeshTriangulation_;
  map::BotCharacterGraphicsItem *selfGraphicsItem_{nullptr};
  map::TrainingAreaGraphicsItem *trainingAreaGraphicsItem_{nullptr};
  std::map<uint32_t, QGraphicsItem*> entityGraphicsItemMap_;
  std::map<uint32_t, QGraphicsItem*> entityMovementGraphicsItemMap_;
  std::vector<QGraphicsLineItem*> walkingPathItems_;

  std::map<sro::scalar_types::EntityGlobalId, std::unique_ptr<entity_data::Entity>> entityData_;

  QTimer *entityMovementUpdateTimer_{nullptr};

  void initializeUi();
  QPixmap parseRegionMinimapPixmapFromPk2(sro::pk2::Pk2ReaderModern &pk2Reader, sro::Sector xSector, sro::Sector ySector);
  QPixmap loadDdjAsQPixmap(sro::pk2::Pk2ReaderModern &pk2Reader, const std::string &path);
  void initializeMap();
  void loadNavmeshIntoScene();
  void connectMainControls();
  void connectTabWidget();
  void connectBotBroadcastMessages();
  void connectConfigControls();
  void connectPacketInjection();
  void triggerMovementTimer();
  void killMovementTimer();

  void entityMovementTimerTriggered();

  void injectPacket(proto::request::PacketToInject::Direction packetDirection, const uint16_t opcode, std::string actualBytes);
  void updateItemList(ItemListWidget *itemListWidget, uint8_t slotIndex, uint16_t quantity, std::optional<std::string> itemName);
  void updateGoldLabel(QLabel *label, uint64_t goldAmount);
  void updateDisplayedPosition(const sro::Position &position);
  void updateDisplayedAngle(qreal angle);
  QPointF sroPositionToMapPosition(const sro::Position &position) const;
  void updateEntityDisplayedPosition(sro::scalar_types::EntityGlobalId globalId, const sro::Position &position, const std::optional<sro::Position> destination = std::nullopt);
  bool haveEntity(sro::scalar_types::EntityGlobalId globalId);

  template<typename EntityType>
  EntityType& getEntity(sro::scalar_types::EntityGlobalId globalId) {
    const auto it = entityData_.find(globalId);
    if (it == entityData_.end()) {
      throw std::runtime_error("No tracking requested entity");
    }
    return dynamic_cast<EntityType&>(*it->second.get());
  }

  template<typename EntityType>
  EntityType* tryGetEntity(sro::scalar_types::EntityGlobalId globalId) {
    const auto it = entityData_.find(globalId);
    if (it == entityData_.end()) {
      return nullptr;
    }
    return dynamic_cast<EntityType*>(it->second.get());
  }

private slots:
  // UI actions
  void startTrainingButtonClicked();
  void stopTrainingButtonClicked();

  void addToDataButtonClicked();
  void injectPacketButtonClicked();
  void reinjectSelectedPackets();
  void clearPackets();

  void timerTriggered();

public slots:
  // Bot updates
  void onLaunch();
  void onCharacterSpawn();
  void onCharacterHpUpdateChanged(uint32_t currentHp);
  void onCharacterMpUpdateChanged(uint32_t currentMp);
  void onCharacterMaxHpMpUpdateChanged(uint32_t maxHp, uint32_t maxMp);
  void onCharacterLevelUpdate(int32_t level, int64_t expRequired);
  void onCharacterExperienceUpdate(uint64_t currentExperience, uint32_t currentSpExperience);
  void onCharacterSpUpdate(uint32_t skillPoints);
  void onCharacterNameUpdate(const std::string &name);
  void onInventoryGoldAmountUpdate(uint64_t goldAmount);
  void onStorageGoldAmountUpdate(uint64_t goldAmount);
  void onGuildStorageGoldAmountUpdate(uint64_t goldAmount);
  void onCharacterPositionChanged(sro::Position currentPosition);
  void onCharacterMovementBeganToDest(sro::Position currentPosition, sro::Position destinationPosition, float speed);
  void onCharacterMovementBeganTowardAngle(sro::Position currentPosition, uint16_t movementAngle, float speed);
  void onCharacterMovementEnded(sro::Position position);
  void onCharacterNotMovingAngleChanged(sro::Angle angle);
  void onRegionNameUpdate(const std::string &regionName);
  void onCharacterInventoryItemUpdate(uint8_t slotIndex, uint16_t quantity, std::optional<std::string> itemName);
  void onAvatarInventoryItemUpdate(uint8_t slotIndex, uint16_t quantity, std::optional<std::string> itemName);
  void onCosInventoryItemUpdate(uint8_t slotIndex, uint16_t quantity, std::optional<std::string> itemName);
  void onStorageItemUpdate(uint8_t slotIndex, uint16_t quantity, std::optional<std::string> itemName);
  void onGuildStorageItemUpdate(uint8_t slotIndex, uint16_t quantity, std::optional<std::string> itemName);
  void onEntitySpawned(uint32_t globalId, sro::Position position, proto::entity::Entity entityData);
  void onEntityDespawned(uint32_t globalId);
  void onEntityPositionChanged(sro::scalar_types::EntityGlobalId globalId, sro::Position position);
  void onEntityMovementBeganToDest(sro::scalar_types::EntityGlobalId globalId, sro::Position currentPosition, sro::Position destinationPosition, float speed);
  void onEntityMovementBeganTowardAngle(sro::scalar_types::EntityGlobalId globalId, sro::Position currentPosition, uint16_t movementAngle, float speed);
  void onEntityMovementEnded(sro::scalar_types::EntityGlobalId globalId, sro::Position position);
  void onEntityLifeStateChanged(sro::scalar_types::EntityGlobalId globalId, sro::entity::LifeState lifeState);
  void onTrainingAreaCircleSet(sro::Position center, float radius);
  void onTrainingAreaReset();
  void onStateMachineCreated(std::string name);
  void onStateMachineDestroyed();
  void onWalkingPathUpdated(std::vector<sro::Position> waypoints);
};

#endif // MAINWINDOW_H
