#include "mainWindow.hpp"
#include "./ui_mainwindow.h"

#include <absl/log/log.h>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent), ui(new Ui::MainWindow) {
  ui->setupUi(this);
}

MainWindow::~MainWindow() {
  delete ui;
}

void MainWindow::setBot(Hyperbot &&bot) {
  LOG(INFO) << "Set bot";
  hyperbot_ = std::move(bot);
}