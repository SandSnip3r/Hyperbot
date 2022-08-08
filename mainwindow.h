#ifndef MAINWINDOW_H
#define MAINWINDOW_H

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
private slots:
  void startTrainingButtonClicked();
  void stopTrainingButtonClicked();
};

//   void injectPacket(request::PacketToInject::Direction packetDirection, const uint16_t opcode, std::string actualBytes);

// public slots:
//   void onEventHandlerConnected();
//   void onMessage1Received(const std::string &str);
//   void onMessage2Received(const int32_t val);
//   void onMessage3Received(const double val);
//   void onVitalsChanged(const broadcast::HpMpUpdate &hpMpUpdate);
// private slots:
//   void onInjectPacketButtonClicked();
//   void onReinjectSelectedPackets();
//   void onClearPackets();
// };
#endif // MAINWINDOW_H
