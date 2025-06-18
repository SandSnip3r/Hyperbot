#include "checkpointWidget.hpp"
#include "hyperbotConnect.hpp"
#include "mainWindow.hpp"
#include "./ui_mainwindow.h"

#include <silkroad_lib/pk2/gameData.hpp>

#include <QtCharts/QChart>
#include <QtCharts/QLineSeries>
#include <QVBoxLayout>
#include <QTabWidget>

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
  dashboardTabs_ = ui->dashboardTabs;
  dashboardTabs_->setTabsClosable(true);
  dashboardWidget_ = new DashboardWidget(gameData_, this);
  dashboardTabs_->addTab(dashboardWidget_, tr("Fleet"));
  dashboards_.append(dashboardWidget_);
  connectDashboard(dashboardWidget_);
  connect(dashboardTabs_, &QTabWidget::tabCloseRequested, this,
          &MainWindow::closeDashboardTab);
  connect(ui->newTabButton, &QPushButton::clicked, this,
          &MainWindow::addDashboardTab);
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
  connectDashboard(dashboardWidget_);

  // TODO: Organize this better
  connect(&hyperbot_, &Hyperbot::plotData, this, &MainWindow::addDataPoint);
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

void MainWindow::connectDashboard(DashboardWidget *widget) {
  connect(&hyperbot_, &Hyperbot::disconnected, widget,
          &DashboardWidget::clearStatusTable);
  connect(&hyperbot_, &Hyperbot::connected, widget,
          &DashboardWidget::onHyperbotConnected);
  connect(&hyperbot_, &Hyperbot::characterStatusReceived, widget,
          &DashboardWidget::onCharacterStatusReceived);
  connect(&hyperbot_, &Hyperbot::activeStateMachineReceived, widget,
          &DashboardWidget::onActiveStateMachine);
  connect(&hyperbot_, &Hyperbot::skillCooldownsReceived, widget,
          &DashboardWidget::onSkillCooldowns);
}

void MainWindow::addDashboardTab() {
  int index = dashboardTabs_->count() + 1;
  DashboardWidget *widget = new DashboardWidget(gameData_, this);
  dashboards_.append(widget);
  connectDashboard(widget);
  dashboardTabs_->addTab(widget, tr("Tab %1").arg(index));
  dashboardTabs_->setCurrentWidget(widget);
}

void MainWindow::closeDashboardTab(int index) {
  if (index <= 0 || index >= dashboardTabs_->count()) {
    // Prevent closing the fleet tab
    return;
  }
  QWidget *widget = dashboardTabs_->widget(index);
  dashboardTabs_->removeTab(index);
  dashboards_.removeOne(static_cast<DashboardWidget *>(widget));
  widget->deleteLater();
}
