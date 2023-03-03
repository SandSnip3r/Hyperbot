#include "itemListWidgetItem.hpp"
#include "mainwindow.h"
#include "map/characterGraphicsItem.hpp"
#include "map/itemGraphicsItem.hpp"
#include "packetListWidgetItem.hpp"
#include "regionGraphicsItem.hpp"
#include "./ui_mainwindow.h"

#include "proto/config.pb.h"

#include <silkroad_lib/game_constants.h>
#include <silkroad_lib/position_math.h>
#include <silkroad_lib/pk2/pk2ReaderModern.h>
#include <silkroad_lib/pk2/navmeshParser.h>

#include <gli/convert.hpp>
#include <gli/format.hpp>
#include <gli/load_dds.hpp>
#include <gli/sampler2d.hpp>

#include <QDir>
#include <QImage>
#include <QMessageBox>

const std::filesystem::path MainWindow::kSilkroadPath_{"C:/Users/Victor/Documents/Development/Daxter Silkroad server files/Silkroad Client"};

namespace {
QImage convertTexture2dToQImage(const gli::texture2d &texture2d);
} // anonymous namespace

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent), ui(new Ui::MainWindow) {
  ui->setupUi(this);
  // TODO: At this point, we assume that we already know our PK2 path and have cached the necessary stuff
  //  In the future, we will need to be able to set the PK2 path after construction of this object
  initializeUi();
  initializeMap();

  connectMainControls();
  connectTabWidget();
  connectBotBroadcastMessages();

  // Start bot connection
  // EventHandler is a subscriber to what the bot publishes
  eventHandler_.runAsync();

  // Requester is a req/rep socket to the bot to cause actions
  requester_.connect();
  // temp();

  // TODO: Reorganize
  // Start timer to update entities positions
  entityMovementUpdateTimer_ = new QTimer(this);
  connect(entityMovementUpdateTimer_, &QTimer::timeout, this, &MainWindow::entityMovementTimerTriggered);
  entityMovementUpdateTimer_->setInterval(1000/60.0);
  entityMovementUpdateTimer_->start();
}

void MainWindow::temp() {
  const std::string kCharacterName{"IP_Man"};
  const std::string kUsername{"5"};
  const std::string kPassword{"0"};
  proto::config::Config config;
  config.set_client_path("C:\\Users\\Victor\\Documents\\Development\\Daxter Silkroad server files\\Silkroad Client");
  config.set_character_to_login(kCharacterName);
  proto::config::CharacterConfig *characterConfig = config.add_character_configs();
  characterConfig->set_character_name(kCharacterName);
  characterConfig->set_username(kUsername);
  characterConfig->set_password(kPassword);
  proto::config::AutopotionConfig *autopotionConfig = characterConfig->mutable_autopotion_config();
  autopotionConfig->set_hp_threshold(0.90);
  autopotionConfig->set_mp_threshold(0.90);
  autopotionConfig->set_vigor_hp_threshold(0.45);
  autopotionConfig->set_vigor_mp_threshold(0.30);
  requester_.sendConfig(config);
}

MainWindow::~MainWindow() {
  delete ui;
}

void MainWindow::initializeUi() {
  ui->hpProgressBar->setStyleSheet(R"(
    QProgressBar {
      border: 1px solid black;
      border-radius: 2px;
      color: white;
      background-color: #131113;
    }
    QProgressBar::chunk {
      background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #630410, stop: 0.5455 #ff3c52, stop: 1 #9c0010);
    }
  )");
  ui->mpProgressBar->setStyleSheet(R"(
    QProgressBar {
      border: 1px solid black;
      border-radius: 2px;
      color: white;
      background-color: #131113;
    }
    QProgressBar::chunk {
      background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #101c4a, stop: 0.5455 #4a69ce, stop: 1 #182c73);
    }
  )");
  ui->characterSpProgressBar->setStyleSheet(R"(
    QProgressBar {
      border: 1px solid black;
      border-radius: 2px;
      color: white;
      background-color: #131113;
    }
    QProgressBar::chunk {
      background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #e7bd10, stop: 0.2 #ffef73, stop: 1 #a57300);
    }
  )");
  ui->characterExperienceProgressBar->setStyleSheet(R"(
    QProgressBar {
      border: 1px solid black;
      border-radius: 2px;
      color: white;
      background-color: #131113;
    }
    QProgressBar::chunk {
      background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #325a1d, stop: 0.5 #8bca50, stop: 1 #1a4213);
    }
  )");
}

std::optional<QPixmap> MainWindow::parseRegionMinimapPixmapFromPk2(sro::pk2::Pk2ReaderModern &pk2Reader, sro::Sector xSector, sro::Sector ySector) {
  const std::string kMinimapDirectory = "minimap\\";
  const std::string kMiniMapImageFileName = std::to_string(xSector) + "x" + std::to_string(ySector) + ".ddj";
  const std::string kMiniMapImagePath = kMinimapDirectory + kMiniMapImageFileName;
  if (!pk2Reader.hasEntry(kMiniMapImagePath)) {
    return {};
  }
  sro::pk2::PK2Entry miniMapImageEntry = pk2Reader.getEntry(kMiniMapImagePath);
  auto miniMapImageData = pk2Reader.getEntryData(miniMapImageEntry);

  // First 20 bytes are joymax specific
  const auto kJoymaxHeaderSize = 20;
  auto *buffer = miniMapImageData.data() + kJoymaxHeaderSize;
  const char *charBuffer = reinterpret_cast<const char*>(buffer);

  const auto texture = gli::load_dds(charBuffer, miniMapImageData.size() - kJoymaxHeaderSize);
  if (texture.size() == 0) {
    // Check if returned texture is not "empty"
    std::cout << "Couldnt parse minimap image" << std::endl;
    return {};
  }

  auto texture2d = gli::texture2d(texture);
  return QPixmap::fromImage(convertTexture2dToQImage(texture2d));
}

void MainWindow::initializeMap() {
  const auto startTime = std::chrono::high_resolution_clock::now();
  loadNavmeshIntoScene();
  const auto endTime = std::chrono::high_resolution_clock::now();
  std::cout << "Loading the navmesh into the scene " << std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime).count()/1000.0 << "s" << std::endl;

  ui->navmeshGraphicsView->setScene(mapScene_);
  if (!navmeshTriangulation_.has_value()) {
    throw std::runtime_error("Navmesh Triangulation missing");
  }
  ui->navmeshGraphicsView->setWorldNavmesh(*navmeshTriangulation_);
  // connect(navmeshGraphicsView, &NavmeshGraphicsView::setPathStart, this, &MainWindow::setPathStart);
  // connect(navmeshGraphicsView, &NavmeshGraphicsView::setPathGoal, this, &MainWindow::setPathGoal);
  // connect(navmeshGraphicsView, &NavmeshGraphicsView::resetPath, this, &MainWindow::resetPath);
  // connect(navmeshGraphicsView, &NavmeshGraphingulation_);
  // connect(navmeshGraphicscsView::mouseMoved, this, &MainWindow::mouseMoved);
}

void MainWindow::loadNavmeshIntoScene() {
  try {
    const auto kMediaPath = kSilkroadPath_ / "Media.pk2";
    const auto kDataPath = kSilkroadPath_ / "Data.pk2";
    sro::pk2::Pk2ReaderModern pk2MediaReader{kMediaPath};
    sro::pk2::Pk2ReaderModern pk2DataReader{kDataPath};

    // Open the file and build the navmesh
    sro::pk2::NavmeshParser navmeshParser{pk2DataReader};
    navmesh_ = navmeshParser.parseNavmesh();
    navmeshTriangulation_ = sro::navmesh::triangulation::NavmeshTriangulation(*navmesh_);
    for (const auto &regionIdTriangulationPair : navmeshTriangulation_->getNavmeshTriangulationMap()) {
      const auto regionId = regionIdTriangulationPair.first;
      const auto [regionX, regionY] = sro::position_math::sectorsFromWorldRegionId(regionId);
      const auto &regionTriangulation = regionIdTriangulationPair.second;

      // Create new graphics item for this region's navmesh
      const auto &region = navmesh_->getRegion(regionId);
      RegionGraphicsItem *item = new RegionGraphicsItem(*navmesh_, region, regionTriangulation);
      {
        // Set minimap pixmap for region
        const auto minimapPixmap = parseRegionMinimapPixmapFromPk2(pk2MediaReader, regionX, regionY);
        if (minimapPixmap) {
          item->setPixmap(*minimapPixmap);
        } else {
          std::cout << "Couldn't load minimap image for region " << static_cast<int>(regionX) << ',' << static_cast<int>(regionY) << std::endl;
        }
      }
      const auto mapPos = sroPositionToMapPosition({regionId, 0.0, 0.0, sro::game_constants::kRegionHeight});
      item->setPos(mapPos);
      mapScene_->addItem(item);
      // Creating labels takes a while, kick it off in another thread
      std::thread thr(&RegionGraphicsItem::createLabels, item);
      thr.detach();
    }
  } catch (std::exception &ex) {
    // Could not open the file
    QMessageBox msgBox;
    msgBox.setText(QString(tr("Could not load navmesh into scene. The file may have invalid .pk2 format. Error: \""))+ex.what()+"\"");
    msgBox.exec();
  }
}

QPointF MainWindow::sroPositionToMapPosition(const sro::Position &position) const {
  const auto [regionX, regionY] = sro::position_math::sectorsFromWorldRegionId(position.regionId());
  return {sro::game_constants::kRegionWidth  *        regionX  +                                       position.xOffset(),
          sro::game_constants::kRegionHeight * (128 - regionY) + (sro::game_constants::kRegionHeight - position.zOffset())};

}

void MainWindow::connectMainControls() {
  connect(ui->startTrainingButton, &QPushButton::clicked, this, &MainWindow::startTrainingButtonClicked);
  connect(ui->stopTrainingButton, &QPushButton::clicked, this, &MainWindow::stopTrainingButtonClicked);
}

void MainWindow::connectTabWidget() {
  connectPacketInjection();
}

void MainWindow::connectBotBroadcastMessages() {
  connect(&eventHandler_, &EventHandler::characterSpawn, this, &MainWindow::onCharacterSpawn);
  connect(&eventHandler_, &EventHandler::characterHpUpdateChanged, this, &MainWindow::onCharacterHpUpdateChanged);
  connect(&eventHandler_, &EventHandler::characterMpUpdateChanged, this, &MainWindow::onCharacterMpUpdateChanged);
  connect(&eventHandler_, &EventHandler::characterMaxHpMpUpdateChanged, this, &MainWindow::onCharacterMaxHpMpUpdateChanged);
  connect(&eventHandler_, &EventHandler::characterLevelUpdate, this, &MainWindow::onCharacterLevelUpdate);
  connect(&eventHandler_, &EventHandler::characterExperienceUpdate, this, &MainWindow::onCharacterExperienceUpdate);
  connect(&eventHandler_, &EventHandler::characterSpUpdate, this, &MainWindow::onCharacterSpUpdate);
  connect(&eventHandler_, &EventHandler::characterNameUpdate, this, &MainWindow::onCharacterNameUpdate);
  connect(&eventHandler_, &EventHandler::inventoryGoldAmountUpdate, this, &MainWindow::onInventoryGoldAmountUpdate);
  connect(&eventHandler_, &EventHandler::storageGoldAmountUpdate, this, &MainWindow::onStorageGoldAmountUpdate);
  connect(&eventHandler_, &EventHandler::guildStorageGoldAmountUpdate, this, &MainWindow::onGuildStorageGoldAmountUpdate);
  connect(&eventHandler_, &EventHandler::characterPositionChanged, this, &MainWindow::onCharacterPositionChanged);
  connect(&eventHandler_, &EventHandler::characterMovementBeganToDest, this, &MainWindow::onCharacterMovementBeganToDest);
  connect(&eventHandler_, &EventHandler::characterMovementBeganTowardAngle, this, &MainWindow::onCharacterMovementBeganTowardAngle);
  connect(&eventHandler_, &EventHandler::characterMovementEnded, this, &MainWindow::onCharacterMovementEnded);
  connect(&eventHandler_, &EventHandler::characterNotMovingAngleChanged, this, &MainWindow::onCharacterNotMovingAngleChanged);
  connect(&eventHandler_, &EventHandler::regionNameUpdate, this, &MainWindow::onRegionNameUpdate);
  connect(&eventHandler_, &EventHandler::characterInventoryItemUpdate, this, &MainWindow::onCharacterInventoryItemUpdate);
  connect(&eventHandler_, &EventHandler::avatarInventoryItemUpdate, this, &MainWindow::onAvatarInventoryItemUpdate);
  connect(&eventHandler_, &EventHandler::cosInventoryItemUpdate, this, &MainWindow::onCosInventoryItemUpdate);
  connect(&eventHandler_, &EventHandler::storageItemUpdate, this, &MainWindow::onStorageItemUpdate);
  connect(&eventHandler_, &EventHandler::guildStorageItemUpdate, this, &MainWindow::onGuildStorageItemUpdate);
  connect(&eventHandler_, &EventHandler::entitySpawned, this, &MainWindow::onEntitySpawned);
  connect(&eventHandler_, &EventHandler::entityDespawned, this, &MainWindow::onEntityDespawned);
  connect(&eventHandler_, &EventHandler::entityPositionChanged, this, &MainWindow::onEntityPositionChanged);
  connect(&eventHandler_, &EventHandler::entityMovementBeganToDest, this, &MainWindow::onEntityMovementBeganToDest);
  connect(&eventHandler_, &EventHandler::entityMovementBeganTowardAngle, this, &MainWindow::onEntityMovementBeganTowardAngle);
  connect(&eventHandler_, &EventHandler::entityMovementEnded, this, &MainWindow::onEntityMovementEnded);
  connect(&eventHandler_, &EventHandler::entityLifeStateChanged, this, &MainWindow::onEntityLifeStateChanged);
  connect(&eventHandler_, &EventHandler::trainingAreaCircleSet, this, &MainWindow::onTrainingAreaCircleSet);
  connect(&eventHandler_, &EventHandler::trainingAreaReset, this, &MainWindow::onTrainingAreaReset);
  connect(&eventHandler_, &EventHandler::stateMachineCreated, this, &MainWindow::onStateMachineCreated);
  connect(&eventHandler_, &EventHandler::stateMachineDestroyed, this, &MainWindow::onStateMachineDestroyed);
}

void MainWindow::connectPacketInjection() {
  // Packet injection tab
  connect(ui->addToDataButton, &QPushButton::clicked, this, &MainWindow::addToDataButtonClicked);
  connect(ui->injectPacketButton, &QPushButton::clicked, this, &MainWindow::injectPacketButtonClicked);
  connect(ui->injectedPacketListWidget, &ReinjectablePacketListWidget::reinjectSelectedPackets, this, &MainWindow::reinjectSelectedPackets);
  connect(ui->injectedPacketListWidget, &ReinjectablePacketListWidget::clearPackets, this, &MainWindow::clearPackets);
}

void MainWindow::triggerMovementTimer() {
  // Cleanup previous timer if it exists
  killMovementTimer();
  // Start timer to update position
  movementUpdateTimer_ = new QTimer(this);
  connect(movementUpdateTimer_, &QTimer::timeout, this, &MainWindow::timerTriggered);
  movementUpdateTimer_->setInterval(kPositionRedrawDelayMs);
  movementUpdateTimer_->start();
}

void MainWindow::timerTriggered() {
  if (!selfData_.movement) {
    throw std::runtime_error("Timer triggered, but we're not moving");
  }
  const auto currentTime = std::chrono::high_resolution_clock::now();
  const auto elapsedTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime-selfData_.movement->startTime).count();
  sro::Position currentPosition;
  if (const auto *destPosPtr = std::get_if<entity_data::Movement::kToDestination>(&selfData_.movement->destPosOrAngle)) {
    const auto totalDistanceToTravel = sro::position_math::calculateDistance2d(selfData_.movement->srcPos, *destPosPtr);
    const auto totalSecondsToTravel = totalDistanceToTravel/selfData_.movement->speed;
    const double fractionTraveled = std::min(1.0, elapsedTimeMs / (totalSecondsToTravel*1000.0));
    currentPosition = sro::position_math::interpolateBetweenPoints(selfData_.movement->srcPos, *destPosPtr, fractionTraveled);
  } else {
    const auto movementAngle = std::get<entity_data::Movement::kTowardAngle>(selfData_.movement->destPosOrAngle);
    const auto totalDistanceTraveled = elapsedTimeMs/1000.0 * selfData_.movement->speed;
    currentPosition = sro::position_math::getNewPositionGivenAngleAndDistance(selfData_.movement->srcPos, movementAngle, totalDistanceTraveled);
  }

  updateDisplayedPosition(currentPosition);
}

void MainWindow::killMovementTimer() {
  if (movementUpdateTimer_ != nullptr) {
    delete movementUpdateTimer_;
    movementUpdateTimer_ = nullptr;
  }
}

void MainWindow::entityMovementTimerTriggered() {
  for (auto &i : entityData_) {
    const auto *mobileEntity = dynamic_cast<entity_data::MobileEntity*>(i.second.get());
    if (mobileEntity == nullptr) {
      // Not a mobile entity, nothing to do
      continue;
    }
    if (!mobileEntity->movement) {
      // Entity is not moving
      continue;
    }
    const auto &movement = *mobileEntity->movement;
    const auto currentTime = std::chrono::high_resolution_clock::now();
    const auto elapsedTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime-movement.startTime).count();
    sro::Position currentPosition;
    if (const auto *destPosPtr = std::get_if<entity_data::Movement::kToDestination>(&movement.destPosOrAngle)) {
      const auto totalDistanceToTravel = sro::position_math::calculateDistance2d(movement.srcPos, *destPosPtr);
      const auto totalSecondsToTravel = totalDistanceToTravel/movement.speed;
      const double fractionTraveled = std::min(1.0, elapsedTimeMs / (totalSecondsToTravel*1000.0));
      currentPosition = sro::position_math::interpolateBetweenPoints(movement.srcPos, *destPosPtr, fractionTraveled);
    } else {
      const auto movementAngle = std::get<entity_data::Movement::kTowardAngle>(movement.destPosOrAngle);
      const auto totalDistanceTraveled = elapsedTimeMs/1000.0 * movement.speed;
      currentPosition = sro::position_math::getNewPositionGivenAngleAndDistance(movement.srcPos, movementAngle, totalDistanceTraveled);
    }

    updateEntityDisplayedPosition(i.first, currentPosition);
  }
}


void MainWindow::injectPacket(request::PacketToInject::Direction packetDirection, const uint16_t opcode, std::string actualBytes) {
  requester_.injectPacket(packetDirection, opcode, actualBytes);
  PacketListWidgetItem *packet = new PacketListWidgetItem(packetDirection, opcode, actualBytes, ui->injectedPacketListWidget);
  ui->injectedPacketListWidget->addItem(packet);
  ui->injectedPacketListWidget->scrollToBottom();
}

void MainWindow::updateItemList(ItemListWidget *itemListWidget, uint8_t slotIndex, uint16_t quantity, std::optional<std::string> itemName) {
  if (quantity == 0) {
    // Remove item
    itemListWidget->removeItem(slotIndex);
  } else {
    // Added or changed item
    if (!itemName) {
      throw std::runtime_error("Received an item without a name");
    }
    ItemListWidgetItem *item = new ItemListWidgetItem(slotIndex, quantity, *itemName);
    itemListWidget->addItem(item);
  }
}

void MainWindow::updateGoldLabel(QLabel *label, uint64_t goldAmount) {
  label->setText(QLocale(QLocale::English).toString(goldAmount));
}

// =================================================================================================================
// ====================================================UI actions===================================================
// =================================================================================================================

// Main controls
void MainWindow::startTrainingButtonClicked() {
  requester_.startTraining();
  ui->startTrainingButton->setEnabled(false);
  ui->stopTrainingButton->setEnabled(true);
}

void MainWindow::stopTrainingButtonClicked() {
  requester_.stopTraining();
  ui->startTrainingButton->setEnabled(true);
  ui->stopTrainingButton->setEnabled(false);

  // TODO: How should we cleanup?
  // Cleanup
  ui->stateListWidget->clear();
}

// Packet injection
void MainWindow::addToDataButtonClicked() {
  // Get type from dropdown (typeComboBox)
  QString dataToAddAsString = ui->addDataLineEdit->text();

  // Get data from line edit (addDataLineEdit)
  QString formattedData;
  if (ui->typeComboBox->currentText() == "float") {
  } else if (ui->typeComboBox->currentText() == "uint32") {
    uint32_t num = dataToAddAsString.toLong();
    // TODO: Check if data already ends with space
    formattedData = QString(" %1 %2 %3 %4").arg(num & 0xFF, 2, 16, QChar('0')).arg((num>>8) & 0xFF, 2, 16, QChar('0')).arg((num>>16) & 0xFF, 2, 16, QChar('0')).arg(num>>24, 2, 16, QChar('0'));
  } else if (ui->typeComboBox->currentText() == "string") {
  }

  // Write formatted data to data text edit (injectPacketDataTextEdit)
  ui->injectPacketDataTextEdit->appendPlainText(formattedData);
}

void MainWindow::injectPacketButtonClicked() {
  request::PacketToInject::Direction packetDirection;
  if (ui->packetInjectionToServerRadioButton->isChecked()) {
    packetDirection = request::PacketToInject::kClientToServer;
  } else {
    packetDirection = request::PacketToInject::kServerToClient;
  }
  if (ui->injectPacketOpcodeLineEdit->text().isEmpty()) {
    std::cout << "injectPacketOpcodeLineEdit is empty" << std::endl;
    return;
  }
  if (ui->injectPacketDataTextEdit->toPlainText().isEmpty()) {
    std::cout << "injectPacketDataTextEdit is empty" << std::endl;
    return;
  }
  const uint16_t opcode = ui->injectPacketOpcodeLineEdit->text().toInt(nullptr, 16);
  const auto bytesString = ui->injectPacketDataTextEdit->toPlainText().toStdString();
  std::string actualBytes;
  for (int i=0; i<bytesString.size(); i+=2) {
    // Skip whitespace
    while (i<bytesString.size() && (bytesString[i] == ' ' || bytesString[i] == '\n' || bytesString[i] == '\r' || bytesString[i] == '\t')) {
      ++i;
    }
    if (i == bytesString.size()) {
      // Had trailing whitespace
      break;
    }

    const std::string byteString = bytesString.substr(i, 2);
    const uint8_t byte = std::stoi(byteString, nullptr, 16);
    actualBytes.push_back(byte);
  }

  injectPacket(packetDirection, opcode, actualBytes);
}


void MainWindow::reinjectSelectedPackets() {
  for (int i=0; i<ui->injectedPacketListWidget->count(); ++i) {
    auto *item = ui->injectedPacketListWidget->item(i);
    if (item->isSelected()) {
      auto *ptr = dynamic_cast<PacketListWidgetItem*>(item);
      if (!ptr) {
        continue;
      }
      injectPacket(ptr->direction(), ptr->opcode(), ptr->data());
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
}

void MainWindow::clearPackets() {
  ui->injectedPacketListWidget->clear();
}

// =================================================================================================================
// ===================================================Bot updates===================================================
// =================================================================================================================

namespace {

qreal sroAngleToQtAngle(const sro::Angle angle) {
  // Silkroad angle 0 means right, Qt 0 means down
  // Silkroad angle is uint16 min to max, Qt is degrees (0-359)
  int int32Angle = std::numeric_limits<uint16_t>::max() - angle;
  // Rotate 270 degrees counterclockwise (90 degrees clockwise)
  int32Angle = (int32Angle + 3 * (1<<14)) % (1<<16);
  // Transform from [0, uint16_max) to [0,360)
  return 360 * static_cast<double>(int32Angle) / std::numeric_limits<uint16_t>::max();
}

qreal calculateAngleFromMovement(const entity_data::Movement &movement) {
  sro::Angle angle;
  if (const auto *destPosPtr = std::get_if<entity_data::Movement::kToDestination>(&movement.destPosOrAngle)) {
    angle = sro::position_math::calculateAngleOfLine(movement.srcPos, *destPosPtr);
  } else {
    angle = std::get<entity_data::Movement::kTowardAngle>(movement.destPosOrAngle);
  }
  return sroAngleToQtAngle(angle);
}

} // anonymous namespace

void MainWindow::onCharacterSpawn() {
  // Reset item list
  ui->characterInventoryListWidget->clear();
  ui->avatarInventoryListWidget->clear();
}

void MainWindow::onCharacterHpUpdateChanged(uint32_t currentHp) {
  selfData_.currentHp = currentHp;
  if (selfData_.currentHp > ui->hpProgressBar->maximum() && ui->hpProgressBar->maximum() != 0) {
    std::cout << "Whoa, setting value to something larger than max" << std::endl;
    std::cout << "Max is " << ui->hpProgressBar->maximum() << " and we're setting it to " << selfData_.currentHp << std::endl;
  }
  ui->hpProgressBar->setValue(selfData_.currentHp);
}

void MainWindow::onCharacterMpUpdateChanged(uint32_t currentMp) {
  selfData_.currentMp = currentMp;
  if (selfData_.currentMp > ui->mpProgressBar->maximum() && ui->mpProgressBar->maximum() != 0) {
    std::cout << "Whoa, setting value to something larger than max" << std::endl;
    std::cout << "Max is " << ui->mpProgressBar->maximum() << " and we're setting it to " << selfData_.currentMp << std::endl;
  }
  ui->mpProgressBar->setValue(selfData_.currentMp);
}

void MainWindow::onCharacterMaxHpMpUpdateChanged(uint32_t maxHp, uint32_t maxMp) {
  selfData_.maxHp = maxHp;
  selfData_.maxMp = maxMp;
  // Overflow in a progress bar is undesireable. If we get a new max value, we will make sure that the current value reflects that
  // Need to set max before setting value to avoid potential overflow
  ui->hpProgressBar->setMaximum(*selfData_.maxHp);
  ui->hpProgressBar->setValue(std::min(static_cast<uint32_t>(selfData_.currentHp), *selfData_.maxHp));
  ui->mpProgressBar->setMaximum(*selfData_.maxMp);
  ui->mpProgressBar->setValue(std::min(static_cast<uint32_t>(selfData_.currentMp), *selfData_.maxMp));
}

void MainWindow::onCharacterLevelUpdate(int32_t level, int64_t expRequired) {
  selfData_.expRequired = expRequired;
  ui->characterLevelLabel->setText(QLocale(QLocale::English).toString(level));
  ui->characterExperienceProgressBar->setMaximum(selfData_.expRequired);
}

void MainWindow::onCharacterExperienceUpdate(uint64_t currentExperience, uint32_t currentSpExperience) {
  if (ui->characterSpProgressBar->maximum() == 0) {
    ui->characterSpProgressBar->setMaximum(selfData_.spExpRequired);
  }
  ui->characterExperienceProgressBar->setValue(currentExperience);
  ui->characterSpProgressBar->setValue(currentSpExperience);
}

void MainWindow::onCharacterSpUpdate(uint32_t skillPoints) {
  ui->characterSpLabel->setText(QLocale(QLocale::English).toString(skillPoints));
}

void MainWindow::onCharacterNameUpdate(const std::string &name) {
  ui->characterNameLabel->setText(QString::fromStdString(name));
}

void MainWindow::onInventoryGoldAmountUpdate(uint64_t goldAmount) {
  updateGoldLabel(ui->inventoryGoldAmountLabel, goldAmount);
}

void MainWindow::onStorageGoldAmountUpdate(uint64_t goldAmount) {
  updateGoldLabel(ui->storageGoldAmountLabel, goldAmount);
}

void MainWindow::onGuildStorageGoldAmountUpdate(uint64_t goldAmount) {
  updateGoldLabel(ui->guildStorageGoldAmountLabel, goldAmount);
}

void MainWindow::onCharacterPositionChanged(sro::Position currentPosition) {
  if (selfData_.movement) {
    // We are moving, update where we are
    selfData_.movement->startTime = std::chrono::high_resolution_clock::now();
    selfData_.movement->srcPos = currentPosition;
    // The running movement timer will move the item within the scene
  } else {
    // We are not moving, we need to move the position of the graphics item ourself
    updateDisplayedPosition(currentPosition);
  }
}

void MainWindow::onCharacterMovementBeganToDest(sro::Position currentPosition, sro::Position destinationPosition, float speed) {
  // Save info
  auto &movement = selfData_.movement.emplace();
  movement.speed = speed;
  movement.startTime = std::chrono::high_resolution_clock::now();
  movement.srcPos = currentPosition;
  movement.destPosOrAngle = destinationPosition;
  triggerMovementTimer();
  updateDisplayedAngle(calculateAngleFromMovement(movement));
  updateDisplayedPosition(currentPosition);
}

void MainWindow::updateDisplayedPosition(const sro::Position &position) {
  // Update label
  const auto gameCoordinate = position.toGameCoordinate();
  ui->characterPositionLabel->setText(QString("%1,%2").arg(gameCoordinate.x).arg(gameCoordinate.y));

  // Update map
  if (selfGraphicsItem_ == nullptr) {
    // Dont yet have a position marker for ourself
    selfGraphicsItem_ = new map::BotCharacterGraphicsItem();
    mapScene_->addItem(selfGraphicsItem_);
  }
  auto mapPosition = sroPositionToMapPosition(position);
  selfGraphicsItem_->setPos(mapPosition);
  ui->navmeshGraphicsView->centerOn(mapPosition);
}

void MainWindow::updateDisplayedAngle(qreal angle) {
  // Update map
  if (selfGraphicsItem_ == nullptr) {
    // Dont yet have a position marker for ourself
    selfGraphicsItem_ = new map::BotCharacterGraphicsItem();
    mapScene_->addItem(selfGraphicsItem_);
  }
  selfGraphicsItem_->setAngle(angle);
}

void MainWindow::onCharacterMovementBeganTowardAngle(sro::Position currentPosition, uint16_t movementAngle, float speed) {
  auto &movement = selfData_.movement.emplace();
  movement.speed = speed;
  movement.startTime = std::chrono::high_resolution_clock::now();
  movement.srcPos = currentPosition;
  movement.destPosOrAngle = movementAngle;
  triggerMovementTimer();
  updateDisplayedAngle(calculateAngleFromMovement(movement));
  updateDisplayedPosition(currentPosition);
}

void MainWindow::onCharacterMovementEnded(sro::Position position) {
  killMovementTimer();
  updateDisplayedPosition(position);
}

void MainWindow::onCharacterNotMovingAngleChanged(sro::Angle angle) {
  updateDisplayedAngle(sroAngleToQtAngle(angle));
}

void MainWindow::onRegionNameUpdate(const std::string &regionName) {
  ui->regionNameLabel->setText(QString::fromStdString(regionName));
}

void MainWindow::onCharacterInventoryItemUpdate(uint8_t slotIndex, uint16_t quantity, std::optional<std::string> itemName) {
  updateItemList(ui->characterInventoryListWidget, slotIndex, quantity, itemName);
}

void MainWindow::onAvatarInventoryItemUpdate(uint8_t slotIndex, uint16_t quantity, std::optional<std::string> itemName) {
  updateItemList(ui->avatarInventoryListWidget, slotIndex, quantity, itemName);
}

void MainWindow::onCosInventoryItemUpdate(uint8_t slotIndex, uint16_t quantity, std::optional<std::string> itemName) {
  updateItemList(ui->cosInventoryListWidget, slotIndex, quantity, itemName);
}

void MainWindow::onStorageItemUpdate(uint8_t slotIndex, uint16_t quantity, std::optional<std::string> itemName) {
  updateItemList(ui->storageListWidget, slotIndex, quantity, itemName);
}

void MainWindow::onGuildStorageItemUpdate(uint8_t slotIndex, uint16_t quantity, std::optional<std::string> itemName) {
  updateItemList(ui->guildStorageListWidget, slotIndex, quantity, itemName);
}

void MainWindow::onEntitySpawned(uint32_t globalId, sro::Position position, broadcast::EntityType entityType) {
  std::unique_ptr<entity_data::Entity> entity;
  // Create an entity object
  switch (entityType) {
    case broadcast::EntityType::kSelf:
      throw std::runtime_error("Right now, we dont expect ourself to be spawning");
    case broadcast::EntityType::kCharacter:
    case broadcast::EntityType::kPlayerCharacter:
    case broadcast::EntityType::kNonplayerCharacter:
    case broadcast::EntityType::kMonsterGeneral:
    case broadcast::EntityType::kMonsterChampion:
    case broadcast::EntityType::kMonsterGiant:
    case broadcast::EntityType::kMonsterElite:
    case broadcast::EntityType::kMonsterUnique:
    case broadcast::EntityType::kMonsterPartyGeneral:
    case broadcast::EntityType::kMonsterPartyChampion:
    case broadcast::EntityType::kMonsterPartyGiant:
      entity = std::make_unique<entity_data::Character>();
      break;
    case broadcast::EntityType::kItemCommon:
    case broadcast::EntityType::kItemRare:
    case broadcast::EntityType::kItemSox:
    case broadcast::EntityType::kPortal:
      entity = std::make_unique<entity_data::Entity>();
      break;
    default:
      throw std::runtime_error("There should be no other entity types");
  }
  entity->globalId = globalId;
  entity->entityType = entityType;

  QGraphicsItem *item;
  if (entityType == broadcast::EntityType::kItemCommon ||
      entityType == broadcast::EntityType::kItemRare ||
      entityType == broadcast::EntityType::kItemSox) {
    item = new map::ItemGraphicsItem(entityType);
  } else {
    item = new map::CharacterGraphicsItem(entityType);
  }
  mapScene_->addItem(item);
  auto mapPosition = sroPositionToMapPosition(position);
  item->setPos(mapPosition);
  [](auto *item, const auto type) {
    const std::map<broadcast::EntityType, qreal> entityZValues = {
      {broadcast::EntityType::kPortal, 1},
      {broadcast::EntityType::kCharacter, 1},
      {broadcast::EntityType::kNonplayerCharacter, 2},
      {broadcast::EntityType::kItemCommon, 3},
      {broadcast::EntityType::kItemRare, 4},
      {broadcast::EntityType::kItemSox, 5},
      {broadcast::EntityType::kPlayerCharacter, 6},
      {broadcast::EntityType::kSelf, 7},
      {broadcast::EntityType::kMonsterGeneral, 8},
      {broadcast::EntityType::kMonsterChampion, 9},
      {broadcast::EntityType::kMonsterPartyGeneral, 10},
      {broadcast::EntityType::kMonsterElite, 11},
      {broadcast::EntityType::kMonsterGiant, 12},
      {broadcast::EntityType::kMonsterPartyChampion, 13},
      {broadcast::EntityType::kMonsterPartyGiant, 14},
      {broadcast::EntityType::kMonsterUnique, 15},
    };
    item->setZValue(entityZValues.at(type));
  }(item, entityType);
  if (auto it = entityGraphicsItemMap_.find(globalId); it != entityGraphicsItemMap_.end()) {
    // Already have an item here, delete it, we'll replace it

    // This happens when the bot disconnects and entities are left behind; when the bot reconnects, it will try to spawn entities that we're already tracking
    // TODO: In the future, when the bot disconnects, we should clean up all entities
    delete it->second;
    entityGraphicsItemMap_.erase(it);
  }
  entityGraphicsItemMap_[globalId] = item;

  // Add entity to entityData map
  entityData_.emplace(globalId, std::move(entity));
}

void MainWindow::onEntityDespawned(uint32_t globalId) {
  auto it = entityGraphicsItemMap_.find(globalId);
  if (it == entityGraphicsItemMap_.end()) {
    // It's ok if we werent tracking this item, nothing to do
    return;
  }
  if (it->second != nullptr) {
    delete it->second;
  } else {
    throw std::runtime_error("Entity despawned, but it already holds a nullptr");
  }
  entityGraphicsItemMap_.erase(it);

  // Remove entity from entityData map
  if (auto it = entityData_.find(globalId); it != entityData_.end()) {
    entityData_.erase(it);
  }
}

void MainWindow::onEntityPositionChanged(sro::scalar_types::EntityGlobalId globalId, sro::Position position) {
  if (!haveEntity(globalId)) {
    // Not tracking this entity
    return;
  }
  auto &mobileEntity = getEntity<entity_data::MobileEntity>(globalId);
  if (mobileEntity.movement) {
    // Entity is moving, update where it is
    mobileEntity.movement->startTime = std::chrono::high_resolution_clock::now();
    mobileEntity.movement->srcPos = position;
    // The running movement timer will move the item within the scene
  } else {
    // Entity is not moving, we need to move the position of the graphics item ourself
    auto it2 = entityGraphicsItemMap_.find(globalId);
    if (it2 == entityGraphicsItemMap_.end()) {
      throw std::runtime_error("Position updated for entity which has no graphics item");
    }
    auto mapPosition = sroPositionToMapPosition(position);
    it2->second->setPos(mapPosition);
  }
}

void MainWindow::onEntityMovementBeganToDest(sro::scalar_types::EntityGlobalId globalId, sro::Position currentPosition, sro::Position destinationPosition, float speed) {
  if (!haveEntity(globalId)) {
    // Not tracking this entity
    return;
  }
  auto &mobileEntity = getEntity<entity_data::MobileEntity>(globalId);
  auto &movement = mobileEntity.movement.emplace();
  movement.speed = speed;
  movement.startTime = std::chrono::high_resolution_clock::now();
  movement.srcPos = currentPosition;
  movement.destPosOrAngle = destinationPosition;
  updateEntityDisplayedPosition(globalId, currentPosition);
}

void MainWindow::onEntityMovementBeganTowardAngle(sro::scalar_types::EntityGlobalId globalId, sro::Position currentPosition, uint16_t movementAngle, float speed) {
  if (!haveEntity(globalId)) {
    // Not tracking this entity
    return;
  }
  auto &mobileEntity = getEntity<entity_data::MobileEntity>(globalId);
  auto &movement = mobileEntity.movement.emplace();
  movement.speed = speed;
  movement.startTime = std::chrono::high_resolution_clock::now();
  movement.srcPos = currentPosition;
  movement.destPosOrAngle = movementAngle;
  updateEntityDisplayedPosition(globalId, currentPosition);
}

void MainWindow::onEntityMovementEnded(sro::scalar_types::EntityGlobalId globalId, sro::Position position) {
  // Cancel movement if it exists
  if (!haveEntity(globalId)) {
    // Not tracking this entity
    return;
  }
  auto &mobileEntity = getEntity<entity_data::MobileEntity>(globalId);
  if (mobileEntity.movement) {
    mobileEntity.movement.reset();
  }
  updateEntityDisplayedPosition(globalId, position);
}

void MainWindow::onEntityLifeStateChanged(sro::scalar_types::EntityGlobalId globalId, sro::entity::LifeState lifeState) {
  if (lifeState == sro::entity::LifeState::kDead) {
    auto it = entityGraphicsItemMap_.find(globalId);
    if (it == entityGraphicsItemMap_.end()) {
      // No map item for this entity
      return;
    }
    auto &characterEntityGraphicsItem = dynamic_cast<map::CharacterGraphicsItem&>(*it->second);
    characterEntityGraphicsItem.setDead();
  }
}

void MainWindow::onTrainingAreaCircleSet(sro::Position center, float radius) {
  if (trainingAreaGraphicsItem_) {
    // Already have a training area graphics item
    delete trainingAreaGraphicsItem_;
  }
  trainingAreaGraphicsItem_ = new map::TrainingAreaGraphicsItem(radius);
  mapScene_->addItem(trainingAreaGraphicsItem_);
  auto mapCenterPosition = sroPositionToMapPosition(center);
  trainingAreaGraphicsItem_->setPos(mapCenterPosition);
}

void MainWindow::onTrainingAreaReset() {
  if (trainingAreaGraphicsItem_) {
    delete trainingAreaGraphicsItem_;
    trainingAreaGraphicsItem_= nullptr;
  }
}

void MainWindow::onStateMachineCreated(std::string name) {
  std::cout << "State machine created: " << name << std::endl;
  ui->stateListWidget->addItem(QString(4*ui->stateListWidget->count(), ' ')+QString::fromStdString(name));
}

void MainWindow::onStateMachineDestroyed() {
  std::cout << "State machine destroyed" << std::endl;
  auto *item = ui->stateListWidget->item(ui->stateListWidget->count()-1);
  delete(item);
}

void MainWindow::updateEntityDisplayedPosition(sro::scalar_types::EntityGlobalId globalId, const sro::Position &position) {
  // Update map
  auto it = entityGraphicsItemMap_.find(globalId);
  if (it == entityGraphicsItemMap_.end()) {
    // No map item for this entity
    return;
  }

  auto mapPosition = sroPositionToMapPosition(position);
  it->second->setPos(mapPosition);
}

bool MainWindow::haveEntity(sro::scalar_types::EntityGlobalId globalId) {
  return entityData_.find(globalId) != entityData_.end();
}

namespace {

QImage convertTexture2dToQImage(const gli::texture2d &texture2d) {
  if (texture2d.format() != gli::FORMAT_RGBA_DXT1_UNORM_BLOCK8) {
    return {};
    // TODO: Why does this happen?
    throw std::runtime_error("Trying to convert wrong texture");
  }
  if (texture2d.levels() < 1) {
    throw std::runtime_error("Have texture with no levels");
  }
  const auto &extent2d = texture2d.extent(0);
  QImage image(extent2d.x, extent2d.y, QImage::Format_RGB32);

  gli::extent2d blockExtent;
  {
    gli::extent3d tempExtent = gli::block_extent(gli::FORMAT_RGBA_DXT1_UNORM_BLOCK8);
    blockExtent.x = tempExtent.x;
    blockExtent.y = tempExtent.y;
  }

  gli::extent2d texelCoord;
  gli::extent2d blockCoord;
  gli::extent2d levelExtent = texture2d.extent(0);
  gli::extent2d levelExtentInBlocks = glm::max(gli::extent2d(1, 1), levelExtent / blockExtent);
  for (blockCoord.y = 0, texelCoord.y = 0; blockCoord.y < levelExtentInBlocks.y; ++blockCoord.y, texelCoord.y += blockExtent.y) {
    for (blockCoord.x = 0, texelCoord.x = 0; blockCoord.x < levelExtentInBlocks.x; ++blockCoord.x, texelCoord.x += blockExtent.x) {
      const gli::detail::dxt1_block *dxt1Block = texture2d.data<gli::detail::dxt1_block>(0, 0, 0) + (blockCoord.y * levelExtentInBlocks.x + blockCoord.x);
      const gli::detail::texel_block4x4 decompressedBlock = gli::detail::decompress_dxt1_block(*dxt1Block);

      gli::extent2d decompressedBlockCoord;
      for (decompressedBlockCoord.y = 0; decompressedBlockCoord.y < glm::min(4, levelExtent.y); ++decompressedBlockCoord.y) {
        for (decompressedBlockCoord.x = 0; decompressedBlockCoord.x < glm::min(4, levelExtent.x); ++decompressedBlockCoord.x) {
          const auto resultingCoordinate = texelCoord + decompressedBlockCoord;
          const auto &texel = decompressedBlock.Texel[decompressedBlockCoord.y][decompressedBlockCoord.x];
          image.setPixelColor(resultingCoordinate.x, resultingCoordinate.y, QColor::fromRgbF(texel.r, texel.g, texel.b, texel.a));
        }
      }
    }
  }
  return image;
}

} // anonymous namespace
