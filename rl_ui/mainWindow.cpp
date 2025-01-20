#include "mainWindow.hpp"
#include "./ui_mainwindow.h"

#include <absl/log/log.h>

MainWindow::MainWindow(Hyperbot &hyperbot, QWidget *parent) : QMainWindow(parent), ui(new Ui::MainWindow), hyperbot_(hyperbot) {
  ui->setupUi(this);
}

MainWindow::~MainWindow() {
  delete ui;
}