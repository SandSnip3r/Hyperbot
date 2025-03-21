#include "hyperbotConnect.hpp"
#include "mainWindow.hpp"
#include "./ui_mainwindow.h"

#include <absl/log/log.h>

#include <QStringListModel>
#include <QtCharts/QChart>
#include <QtCharts/QLineSeries>

#include <random>

// Things which I need to plot:
//  - Loss
//  - Reward over the last some amount of time, maybe configurable by user
//  - Total episode reward
//  - Epsilon
//  - Win rate

MainWindow::MainWindow(Config &&config, Hyperbot &hyperbot, QWidget *parent) : QMainWindow(parent), ui(new Ui::MainWindow), config_(std::move(config)), hyperbot_(hyperbot) {
  ui->setupUi(this);
  setWindowTitle(tr("Hyperbot"));
  connectSignals();
}

void MainWindow::showEvent(QShowEvent *event) {
  // Always call base implementation.
  QMainWindow::showEvent(event);

  if (!connectionWindowShown_) {
    connectionWindowShown_ = true;
    connectionWindow_ = new HyperbotConnect(config_, hyperbot_, this);
    connect(connectionWindow_, &QObject::destroyed, this, &MainWindow::connectedToHyperbot);
    this->setEnabled(false);
    connectionWindow_->setEnabled(true);
    connectionWindow_->show();
  }
}

MainWindow::~MainWindow() {
  delete ui;
}

void MainWindow::connectSignals() {
  connect(ui->startTrainingButton, &QPushButton::clicked, &hyperbot_, &Hyperbot::startTraining);
  connect(ui->stopTrainingButton, &QPushButton::clicked, &hyperbot_, &Hyperbot::stopTraining);
  connect(&hyperbot_, &Hyperbot::checkpointListReceived, this, &MainWindow::checkpointListReceived);
}

void MainWindow::testChart() {
  series_ = new QLineSeries;
  series_->append(0,6);
  series_->append(2,4);
  series_->append(3,8);
  series_->append(7,4);
  series_->append(10,5);
  QChart *chart = new QChart;
  chart->legend()->hide();
  chart->addSeries(series_);
  chart->createDefaultAxes();
  chart->setTitle("My title");
  ui->graphWidget->setChart(chart);
  // chart->axes()

  timer_ = new QTimer(this);
  timer_->setInterval(100);
  connect(timer_, &QTimer::timeout, this, &MainWindow::timerTriggered);
  timer_->start();
}

void MainWindow::connectedToHyperbot() {
  connectionWindow_ = nullptr;
  this->setEnabled(true);
  hyperbot_.requestCheckpointList();
}

void MainWindow::timerTriggered() {
  static std::mt19937 gen(std::random_device{}());
  static std::uniform_real_distribution<double> dist(0, 100);
  static int xVal{11};
  int y = dist(gen);
  series_->append(xVal++, y);
  LOG(INFO) << "Trigger";
}

void MainWindow::checkpointListReceived(QStringList list) {
  VLOG(1) << "Received checkpoint list: " << list.join(", ").toStdString();
  QStringListModel *checkpointModel = new QStringListModel(list);
  ui->checkpointsListView->setModel(checkpointModel);
}