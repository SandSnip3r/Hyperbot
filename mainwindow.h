#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "characterData.hpp"
#include "entityGraphicsItem.hpp"
#include "eventHandler.hpp"
#include "itemListWidget.hpp"
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
#include <optional>
#include <string>

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
  static constexpr int kPositionRedrawDelayMs{33};
  Ui::MainWindow *ui;
  zmq::context_t context_;
  EventHandler eventHandler_{context_};
  Requester requester_{context_};
  CharacterData characterData_;
  QTimer *movementUpdateTimer_{nullptr};
  QGraphicsScene *mapScene_{new QGraphicsScene(this)};
  std::optional<sro::navmesh::Navmesh> navmesh_;
  std::optional<sro::navmesh::triangulation::NavmeshTriangulation> navmeshTriangulation_;
  EntityGraphicsItem *entityGraphicsItem_{nullptr};
  std::map<uint32_t, EntityGraphicsItem*> entityGraphicsItemMap_;

  void initializeUi();
  std::optional<QPixmap> parseRegionMinimapPixmapFromPk2(sro::pk2::Pk2ReaderModern &pk2Reader, sro::Sector xSector, sro::Sector ySector);
  void initializeMap();
  void loadNavmeshIntoScene();
  void connectMainControls();
  void connectTabWidget();
  void connectBotBroadcastMessages();
  void connectPacketInjection();
  void triggerMovementTimer();
  void killMovementTimer();

  void injectPacket(request::PacketToInject::Direction packetDirection, const uint16_t opcode, std::string actualBytes);
  void updateItemList(ItemListWidget *itemListWidget, uint8_t slotIndex, uint16_t quantity, std::optional<std::string> itemName);
  void updateGoldLabel(QLabel *label, uint64_t goldAmount);
  void updateDisplayedPosition(const sro::Position &position);
  QPointF sroPositionToMapPosition(const sro::Position &position) const;

private slots:
  // UI actions
  void startTrainingButtonClicked();
  void stopTrainingButtonClicked();

  void injectPacketButtonClicked();
  void reinjectSelectedPackets();
  void clearPackets();

  void timerTriggered();

public slots:
  // Bot updates
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
  void onCharacterMovementBeganToDest(sro::Position currentPosition, sro::Position destinationPosition, float speed);
  void onCharacterMovementBeganTowardAngle(sro::Position currentPosition, uint16_t movementAngle, float speed);
  void onCharacterMovementEnded(sro::Position position);
  void onRegionNameUpdate(const std::string &regionName);
  void onCharacterInventoryItemUpdate(uint8_t slotIndex, uint16_t quantity, std::optional<std::string> itemName);
  void onAvatarInventoryItemUpdate(uint8_t slotIndex, uint16_t quantity, std::optional<std::string> itemName);
  void onCosInventoryItemUpdate(uint8_t slotIndex, uint16_t quantity, std::optional<std::string> itemName);
  void onStorageItemUpdate(uint8_t slotIndex, uint16_t quantity, std::optional<std::string> itemName);
  void onGuildStorageItemUpdate(uint8_t slotIndex, uint16_t quantity, std::optional<std::string> itemName);
  void onEntitySpawned(uint32_t globalId, sro::Position position, sro::entity_types::EntityType entityType);
  void onEntityDespawned(uint32_t globalId);
};

#endif // MAINWINDOW_H
