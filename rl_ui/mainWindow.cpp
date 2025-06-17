#include "checkpointWidget.hpp"
#include "hyperbotConnect.hpp"
#include "mainWindow.hpp"
#include "./ui_mainwindow.h"

#include <silkroad_lib/pk2/gameData.hpp>

#include <QtCharts/QChart>
#include <QtCharts/QLineSeries>
#include <QVBoxLayout>
#include <QDockWidget>
#include <QMimeData>

#include <absl/log/log.h>

#include <random>

// Things which I need to plot:
//  - Loss
//  - Reward over the last some amount of time, maybe configurable by user
//  - Total episode reward
//  - Epsilon
//  - Win rate

MainWindow::MainWindow(Config &&config, Hyperbot &hyperbot,
                       const sro::pk2::GameData &gameData,
                       QWidget *parent)
    : QMainWindow(parent),
      ui(new Ui::MainWindow),
      config_(std::move(config)),
      hyperbot_(hyperbot),
      gameData_(gameData) {
  ui->setupUi(this);
  setAcceptDrops(true);
  ui->checkpointWidget->setHyperbot(hyperbot_);
  ui->graphWidget->chart()->setTitle(tr("Event Queue Size"));
  setWindowTitle(tr("Hyperbot"));
  dashboardWidget_ = new DashboardWidget(gameData_, this);

  fleetDock_ = new QDockWidget(tr("Fleet"), this);
  fleetDock_->setObjectName("fleetDock");
  fleetDock_->setWidget(dashboardWidget_);
  addDockWidget(Qt::LeftDockWidgetArea, fleetDock_);

  checkpointDock_ = new QDockWidget(tr("Checkpoints"), this);
  checkpointDock_->setObjectName("checkpointDock");
  checkpointDock_->setWidget(ui->checkpointWidget);
  addDockWidget(Qt::LeftDockWidgetArea, checkpointDock_);

  chartDock_ = new QDockWidget(tr("Queue Size"), this);
  chartDock_->setObjectName("chartDock");
  chartDock_->setWidget(ui->graphWidget);
  addDockWidget(Qt::RightDockWidgetArea, chartDock_);

  ui->dashboardContainer->setParent(nullptr);
  ui->checkpointWidget->setParent(nullptr);
  ui->graphWidget->setParent(nullptr);
  connectSignals();
  // testChart();
}

void MainWindow::showEvent(QShowEvent *event) {
  // Always call base implementation.
  QMainWindow::showEvent(event);
  showConnectionWindow(tr("Connect to Hyperbot"));
}

MainWindow::~MainWindow() {
  delete ui;
}

void MainWindow::connectSignals() {
  connect(ui->startTrainingButton, &QPushButton::clicked, &hyperbot_, &Hyperbot::startTraining);
  connect(ui->stopTrainingButton, &QPushButton::clicked, &hyperbot_, &Hyperbot::stopTraining);
  connect(&hyperbot_, &Hyperbot::disconnected, this, &MainWindow::onDisconnectedFromHyperbot);
  connect(&hyperbot_, &Hyperbot::disconnected, dashboardWidget_,
          &DashboardWidget::clearStatusTable);
  connect(&hyperbot_, &Hyperbot::connected, dashboardWidget_,
          &DashboardWidget::onHyperbotConnected);

  // TODO: Organize this better
  connect(&hyperbot_, &Hyperbot::plotData, this, &MainWindow::addDataPoint);
  connect(&hyperbot_, &Hyperbot::characterStatusReceived, dashboardWidget_,
          &DashboardWidget::onCharacterStatusReceived);
  connect(&hyperbot_, &Hyperbot::activeStateMachineReceived, dashboardWidget_,
          &DashboardWidget::onActiveStateMachine);
  connect(&hyperbot_, &Hyperbot::skillCooldownsReceived, dashboardWidget_,
          &DashboardWidget::onSkillCooldowns);

  connect(dashboardWidget_, &DashboardWidget::characterDetailRequested, this,
          &MainWindow::openCharacterDetailDock);
  connect(dashboardWidget_, &DashboardWidget::characterDataUpdated, this,
          [this](QString name, CharacterData data) {
            if (detailDocks_.contains(name)) {
              auto *dialog = qobject_cast<CharacterDetailDialog *>(detailDocks_[name]->widget());
              if (dialog) {
                dialog->onCharacterDataUpdated(name, data);
              }
            }
          });
}

void MainWindow::showConnectionWindow(const QString &windowTitle) {
  if (!connectionWindowShown_) {
    connectionWindowShown_ = true;
    connectionWindow_ = new HyperbotConnect(config_, hyperbot_, this);
    connectionWindow_->setWindowTitle(windowTitle);
    connect(connectionWindow_, &QObject::destroyed, this, &MainWindow::onConnectedToHyperbot);
    this->setEnabled(false);
    connectionWindow_->setEnabled(true);
    connectionWindow_->show();
  }
}

void MainWindow::testChart() {
  ui->graphWidget->chart()->setBackgroundRoundness(0);
  ui->graphWidget->chart()->setBackgroundVisible(false);

  ui->graphWidget->addDataPoint({0,6});
  ui->graphWidget->addDataPoint({2,4});
  ui->graphWidget->addDataPoint({3,8});
  ui->graphWidget->addDataPoint({7,4});
  ui->graphWidget->addDataPoint({10,5});
  timer_ = new QTimer(this);
  timer_->setInterval(100);
  connect(timer_, &QTimer::timeout, this, &MainWindow::onTimerTriggered);
  timer_->start();
}

void MainWindow::onConnectedToHyperbot() {
  connectionWindowShown_ = false;
  connectionWindow_ = nullptr;
  this->setEnabled(true);
}

void MainWindow::onDisconnectedFromHyperbot() {
  showConnectionWindow(tr("Reconnect to Hyperbot"));
}

void MainWindow::onTimerTriggered() {
  static std::mt19937 gen(std::random_device{}());
  static std::uniform_real_distribution<double> dist(0, 100);
  static qreal xVal{11};
  qreal y = dist(gen);
  ui->graphWidget->addDataPoint({xVal, y});
  xVal += 1;
}

void MainWindow::addDataPoint(qreal x, qreal y) {
  static int count=0;
  ++count;
  if (count % 1000 == 0) {
    LOG(INFO) << "Adding data point #" << count;
  }
  ui->graphWidget->addDataPoint({x,y});
}

void MainWindow::dragEnterEvent(QDragEnterEvent *event) {
  if (event->mimeData()->hasText()) {
    event->acceptProposedAction();
  }
}

void MainWindow::dropEvent(QDropEvent *event) {
  QString name = event->mimeData()->text();
  openCharacterDetailDock(name);
  event->acceptProposedAction();
}

void MainWindow::openCharacterDetailDock(const QString &name) {
  if (detailDocks_.contains(name)) {
    detailDocks_[name]->raise();
    return;
  }
  QDockWidget *dock = new QDockWidget(name, this);
  dock->setAttribute(Qt::WA_DeleteOnClose);
  CharacterDetailDialog *dialog = new CharacterDetailDialog(gameData_, dock);
  dialog->setCharacterName(name);
  dialog->updateCharacterData(dashboardWidget_->getCharacterData(name));
  connect(dock, &QObject::destroyed, this,
          [this, name]() { detailDocks_.remove(name); });
  connect(dashboardWidget_, &DashboardWidget::characterDataUpdated, dialog,
          &CharacterDetailDialog::onCharacterDataUpdated);
  dock->setWidget(dialog);
  addDockWidget(Qt::RightDockWidgetArea, dock);
  detailDocks_.insert(name, dock);
}
