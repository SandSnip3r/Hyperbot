#include "checkpointWidget.hpp"
#include "hyperbotConnect.hpp"
#include "mainWindow.hpp"
#include "./ui_mainWindow.h"

#include <silkroad_lib/pk2/gameData.hpp>

#include <QtCharts/QChart>
#include <QtCharts/QLineSeries>
#include <QVBoxLayout>

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
  ui->checkpointWidget->setHyperbot(hyperbot_);
  ui->graphWidget->chart()->setTitle(tr("Event Queue Size"));
  setWindowTitle(tr("Hyperbot"));
  dashboardWidget_ = new DashboardWidget(gameData_, this);
  QVBoxLayout *layout = qobject_cast<QVBoxLayout*>(ui->dashboardContainer->layout());
  if (!layout) {
    layout = new QVBoxLayout(ui->dashboardContainer);
  }
  layout->addWidget(dashboardWidget_);
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
  connect(&hyperbot_, &Hyperbot::qValuesReceived, dashboardWidget_,
          &DashboardWidget::onQValues);
  connect(&hyperbot_, &Hyperbot::itemCountReceived, dashboardWidget_,
          &DashboardWidget::onItemCount);
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