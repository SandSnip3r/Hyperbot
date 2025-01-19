#include "hyperbotConnect.hpp"
#include "mainWindow.hpp"
#include "ui_hyperbotconnect.h"

#include <absl/log/log.h>

HyperbotConnect::HyperbotConnect(Config &&config, QWidget *parent) : QMainWindow(parent), ui(new Ui::HyperbotConnect), config_(config) {
  ui->setupUi(this);
  ui->ipAddressLineEdit->setText(QString::fromStdString(config.proto().ip_address()));
  ui->portLineEdit->setText(QString::number(config.proto().port()));
  connect(ui->connectButton, &QPushButton::clicked, this, &HyperbotConnect::connectClicked);
  connect(ui->cancelButton, &QPushButton::clicked, this, &HyperbotConnect::close);
}

HyperbotConnect::~HyperbotConnect() {
  delete ui;
}

void HyperbotConnect::connectClicked() {
  LOG(INFO) << "Connect button clicked";
  std::string ipAddress = ui->ipAddressLineEdit->text().toStdString();
  int32_t port = ui->portLineEdit->text().toInt();
  config_.proto().set_ip_address(ipAddress);
  config_.proto().set_port(port);
  config_.save();

  // Try to connect to the bot.
  Hyperbot bot;
  bot.connect(ipAddress, port);
  MainWindow *mw = new MainWindow();
  mw->setBot(std::move(bot));
  mw->show();
  close();
}