#include "checkpointWidget.hpp"
#include "hyperbotConnect.hpp"
#include "mainWindow.hpp"
#include "./ui_mainwindow.h"

#include <absl/log/log.h>

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
  ui->checkpointWidget->setHyperbot(hyperbot_);
  setWindowTitle(tr("Hyperbot"));
  connectSignals();
  testChart();
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
  series_ = new QLineSeries;
  QChart *chart = new QChart;
  ui->graphWidget->setRenderHint(QPainter::Antialiasing);
  chart->legend()->hide();
  chart->addSeries(series_);
  chart->createDefaultAxes();
  chart->setTitle("My title");
  chart->setBackgroundRoundness(0);
  chart->setBackgroundVisible(false);
  ui->graphWidget->setChart(chart);

  addDataPoint(0,6);
  addDataPoint(2,4);
  addDataPoint(3,8);
  addDataPoint(7,4);
  addDataPoint(10,5);
  timer_ = new QTimer(this);
  timer_->setInterval(100);
  connect(timer_, &QTimer::timeout, this, &MainWindow::onTimerTriggered);
  timer_->start();
}

void MainWindow::onConnectedToHyperbot() {
  connectionWindowShown_ = false;
  connectionWindow_ = nullptr;
  this->setEnabled(true);
  hyperbot_.requestCheckpointList();
}

void MainWindow::onDisconnectedFromHyperbot() {
  showConnectionWindow(tr("Reconnect to Hyperbot"));
}

void MainWindow::onTimerTriggered() {
  static std::mt19937 gen(std::random_device{}());
  static std::uniform_real_distribution<double> dist(0, 100);
  static int xVal{11};
  int y = dist(gen);
  addDataPoint(xVal, y);
  xVal++;
  if (xVal > 100) {
    timer_->stop();
  }
}

void MainWindow::addDataPoint(float x, float y) {
  series_->append(x, y);
  QChart *chart = ui->graphWidget->chart();
  if (chart) {
    bool scaleChanged = false;
    if (x < minX_) {
      minX_ = x;
      scaleChanged = true;
    }
    if (x > maxX_) {
      maxX_ = x;
      scaleChanged = true;
    }
    if (scaleChanged) {
      QAbstractAxis *axisX = chart->axes(Qt::Horizontal).first();
      if (axisX) {
        axisX->setRange(minX_, maxX_);
      }
    }
    scaleChanged = false;
    if (y < minY_) {
      minY_ = y;
      scaleChanged = true;
    }
    if (y > maxY_) {
      maxY_ = y;
      scaleChanged = true;
    }
    if (scaleChanged) {
      QAbstractAxis *axisY = chart->axes(Qt::Vertical).first();
      if (axisY) {
        axisY->setRange(minY_, maxY_);
      }
    }
  }

}