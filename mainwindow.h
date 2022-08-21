#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "characterData.hpp"
#include "eventHandler.hpp"
#include "requester.hpp"

#include <zmq.hpp>

#include <QMainWindow>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
  Q_OBJECT
public:
  MainWindow(QWidget *parent = nullptr);
  ~MainWindow();

private:
  Ui::MainWindow *ui;
  zmq::context_t context_;
  EventHandler eventHandler_{context_};
  Requester requester_{context_};
  CharacterData characterData_;

  void initializeUi();
  void connectMainControls();
  void connectTabWidget();
  void connectBotBroadcastMessages();
  void connectPacketInjection();

  void injectPacket(request::PacketToInject::Direction packetDirection, const uint16_t opcode, std::string actualBytes);
private slots:
  // UI actions
  void startTrainingButtonClicked();
  void stopTrainingButtonClicked();

  void injectPacketButtonClicked();
  void reinjectSelectedPackets();
  void clearPackets();

public slots:
  // Bot updates
  void onCharacterHpUpdateChanged(uint32_t currentHp);
  void onCharacterMpUpdateChanged(uint32_t currentMp);
  void onCharacterMaxHpMpUpdateChanged(uint32_t maxHp, uint32_t maxMp);
  void onCharacterLevelUpdate(int32_t level, int64_t expRequired);
  void onCharacterExperienceUpdate(uint64_t currentExperience, uint32_t currentSpExperience);
  void onCharacterSpUpdate(uint32_t skillPoints);
  void onCharacterNameUpdate(const std::string &name);
  void onInventoryGoldAmountUpdate(uint64_t goldAmount);
  void onRegionNameUpdate(const std::string &regionName);
};

#endif // MAINWINDOW_H
