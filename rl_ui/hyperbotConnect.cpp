#include "hyperbotConnect.hpp"
#include "mainWindow.hpp"
#include "ui_hyperbotconnect.h"

#include <absl/log/log.h>

HyperbotConnect::HyperbotConnect(Config &&config, Hyperbot &hyperbot, QWidget *parent) : QMainWindow(parent), ui(new Ui::HyperbotConnect), config_(config), hyperbot_(hyperbot) {
  ui->setupUi(this);
  ui->ipAddressLineEdit->setText(QString::fromStdString(config.proto().ip_address()));
  ui->portLineEdit->setText(QString::number(config.proto().port()));
  connect(ui->connectButton, &QPushButton::clicked, this, &HyperbotConnect::connectClicked);
  connect(ui->cancelButton, &QPushButton::clicked, this, &HyperbotConnect::close);
}

HyperbotConnect::~HyperbotConnect() {
  std::cout << "Deconstructing HyperbotConnect" << std::endl;
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
  bool connected = hyperbot_.connect(ipAddress, port);
  if (!connected) {
    LOG(WARNING) << "Failed to connect to bot";
    return;
  }
  MainWindow *mw = new MainWindow(hyperbot_);
  mw->show();
  close();
}