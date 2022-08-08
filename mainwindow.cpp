#include "mainwindow.h"
#include "./ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent), ui(new Ui::MainWindow) {
  ui->setupUi(this);

  // Connect buttons of main page
  connect(ui->startTrainingButton, &QPushButton::clicked, this, &MainWindow::startTrainingButtonClicked);
  connect(ui->stopTrainingButton, &QPushButton::clicked, this, &MainWindow::stopTrainingButtonClicked);

  // Start bot connection
  // EventHandler is a subscriber to what the bot publishes
  eventHandler_.runAsync();

  // Requester is a req/rep socket to the bot to cause actions
  requester_.connect();
}

MainWindow::~MainWindow() {
  delete ui;
}

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
