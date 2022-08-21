#include "mainwindow.h"
#include "packetListWidgetItem.hpp"
#include "./ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent), ui(new Ui::MainWindow) {
  ui->setupUi(this);
  initializeUi();

  connectMainControls();
  connectTabWidget();
  connectBotBroadcastMessages();

  // Start bot connection
  // EventHandler is a subscriber to what the bot publishes
  eventHandler_.runAsync();

  // Requester is a req/rep socket to the bot to cause actions
  requester_.connect();
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

void MainWindow::connectMainControls() {
  connect(ui->startTrainingButton, &QPushButton::clicked, this, &MainWindow::startTrainingButtonClicked);
  connect(ui->stopTrainingButton, &QPushButton::clicked, this, &MainWindow::stopTrainingButtonClicked);
}

void MainWindow::connectTabWidget() {
  connectPacketInjection();
}

void MainWindow::connectBotBroadcastMessages() {
  connect(&eventHandler_, &EventHandler::characterHpUpdateChanged, this, &MainWindow::onCharacterHpUpdateChanged);
  connect(&eventHandler_, &EventHandler::characterMpUpdateChanged, this, &MainWindow::onCharacterMpUpdateChanged);
  connect(&eventHandler_, &EventHandler::characterMaxHpMpUpdateChanged, this, &MainWindow::onCharacterMaxHpMpUpdateChanged);
  connect(&eventHandler_, &EventHandler::characterLevelUpdate, this, &MainWindow::onCharacterLevelUpdate);
  connect(&eventHandler_, &EventHandler::characterExperienceUpdate, this, &MainWindow::onCharacterExperienceUpdate);
  connect(&eventHandler_, &EventHandler::characterSpUpdate, this, &MainWindow::onCharacterSpUpdate);
  connect(&eventHandler_, &EventHandler::characterNameUpdate, this, &MainWindow::onCharacterNameUpdate);
  connect(&eventHandler_, &EventHandler::inventoryGoldAmountUpdate, this, &MainWindow::onInventoryGoldAmountUpdate);
  connect(&eventHandler_, &EventHandler::regionNameUpdate, this, &MainWindow::onRegionNameUpdate);
}

void MainWindow::connectPacketInjection() {
  // Packet injection tab
  connect(ui->injectPacketButton, &QPushButton::clicked, this, &MainWindow::injectPacketButtonClicked);
  connect(ui->injectedPacketListWidget, &ReinjectablePacketListWidget::reinjectSelectedPackets, this, &MainWindow::reinjectSelectedPackets);
  connect(ui->injectedPacketListWidget, &ReinjectablePacketListWidget::clearPackets, this, &MainWindow::clearPackets);
}

MainWindow::~MainWindow() {
  delete ui;
}

void MainWindow::injectPacket(request::PacketToInject::Direction packetDirection, const uint16_t opcode, std::string actualBytes) {
  requester_.injectPacket(packetDirection, opcode, actualBytes);
  PacketListWidgetItem *packet = new PacketListWidgetItem(packetDirection, opcode, actualBytes, ui->injectedPacketListWidget);
  ui->injectedPacketListWidget->addItem(packet);
  ui->injectedPacketListWidget->scrollToBottom();
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
}

// Packet injection
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

void MainWindow::onCharacterHpUpdateChanged(uint32_t currentHp) {
  characterData_.currentHp = currentHp;
  if (characterData_.currentHp > ui->hpProgressBar->maximum() && ui->hpProgressBar->maximum() != 0) {
    std::cout << "Whoa, setting value to something larger than max" << std::endl;
    std::cout << "Max is " << ui->hpProgressBar->maximum() << " and we're setting it to " << characterData_.currentHp << std::endl;
  }
  ui->hpProgressBar->setValue(characterData_.currentHp);
}

void MainWindow::onCharacterMpUpdateChanged(uint32_t currentMp) {
  characterData_.currentMp = currentMp;
  if (characterData_.currentMp > ui->mpProgressBar->maximum() && ui->mpProgressBar->maximum() != 0) {
    std::cout << "Whoa, setting value to something larger than max" << std::endl;
    std::cout << "Max is " << ui->mpProgressBar->maximum() << " and we're setting it to " << characterData_.currentMp << std::endl;
  }
  ui->mpProgressBar->setValue(characterData_.currentMp);
}

void MainWindow::onCharacterMaxHpMpUpdateChanged(uint32_t maxHp, uint32_t maxMp) {
  characterData_.maxHp = maxHp;
  characterData_.maxMp = maxMp;
  // Overflow in a progress bar is undesireable. If we get a new max value, we will make sure that the current value reflects that
  // Need to set max before setting value to avoid potential overflow
  ui->hpProgressBar->setMaximum(*characterData_.maxHp);
  ui->hpProgressBar->setValue(std::min(static_cast<uint32_t>(characterData_.currentHp), *characterData_.maxHp));
  ui->mpProgressBar->setMaximum(*characterData_.maxMp);
  ui->mpProgressBar->setValue(std::min(static_cast<uint32_t>(characterData_.currentMp), *characterData_.maxMp));
}

void MainWindow::onCharacterLevelUpdate(int32_t level, int64_t expRequired) {
  characterData_.expRequired = expRequired;
  ui->characterLevelLabel->setText(QString::number(level));
  ui->characterExperienceProgressBar->setMaximum(characterData_.expRequired);
}

void MainWindow::onCharacterExperienceUpdate(uint64_t currentExperience, uint32_t currentSpExperience) {
  if (ui->characterSpProgressBar->maximum() == 0) {
    ui->characterSpProgressBar->setMaximum(characterData_.spExpRequired);
  }
  ui->characterExperienceProgressBar->setValue(currentExperience);
  ui->characterSpProgressBar->setValue(currentSpExperience);
}

void MainWindow::onCharacterSpUpdate(uint32_t skillPoints) {
  ui->characterSpLabel->setText(QString(tr("%1")).arg(skillPoints));
}

void MainWindow::onCharacterNameUpdate(const std::string &name) {
  ui->characterNameLabel->setText(QString::fromStdString(name));
}

void MainWindow::onInventoryGoldAmountUpdate(uint64_t goldAmount) {
  ui->inventoryGoldAmountLabel->setText(QString::number(goldAmount));
}

void MainWindow::onRegionNameUpdate(const std::string &regionName) {
  ui->regionNameLabel->setText(QString::fromStdString(regionName));
}
