#include "hyperbotConnect.hpp"
#include "mainWindow.hpp"
#include "ui_hyperbotconnect.h"

#include <absl/log/log.h>
#include <absl/log/log_sink_registry.h>
#include <absl/strings/str_format.h>
#include <absl/time/time.h>

#include <QDateTime>

namespace {

// Helper function to convert absl::Time to QDateTime in local time
QDateTime abslTimeToQDateTime(absl::Time absl_time) {
  // Abseil's UnixEpoch is 1970-01-01T00:00:00Z
  absl::Time unix_epoch = absl::UnixEpoch();
  absl::Duration delta = absl_time - unix_epoch;

  // Get total seconds and nanoseconds since Unix epoch
  int64_t seconds = absl::ToInt64Seconds(delta);
  int64_t nanos = absl::ToInt64Nanoseconds(delta) % 1000000000;

  // Create QDateTime from seconds since epoch
  QDateTime qdt = QDateTime::fromSecsSinceEpoch(seconds, Qt::LocalTime);

  // Add remaining milliseconds (nanos / 1,000,000)
  qdt = qdt.addMSecs(static_cast<int>(nanos / 1000000));

  return qdt;
}

} // anonymous namespace

namespace internal {

void MyLogSink::Send(const absl::LogEntry& entry) {
  if (entry.verbosity() >= 0) {
    // Do not write vlogs to the UI.
    return;
  }

  absl::Time timestamp = entry.timestamp();
  QDateTime qdt = abslTimeToQDateTime(timestamp);

  // Format the QDateTime as desired
  QString formatted_time = qdt.toString("HH:mm:ss");
  QString logMessage = QString::fromStdString(absl::StrFormat("[%s]: %s", formatted_time.toStdString(), entry.text_message()));
  QMetaObject::invokeMethod(loggingTextEdit_, "append", Qt::AutoConnection, Q_ARG(QString, logMessage));
}

} // namespace internal

HyperbotConnect::HyperbotConnect(Config &config, Hyperbot &hyperbot, QWidget *parent) : QMainWindow(parent), ui(new Ui::HyperbotConnect), config_(config), hyperbot_(hyperbot) {
  setAttribute(Qt::WA_DeleteOnClose);
  ui->setupUi(this);
  this->setWindowFlags(Qt::Window |
                       Qt::WindowMinimizeButtonHint |
                       Qt::WindowMaximizeButtonHint |
                       Qt::CustomizeWindowHint);
  registerLogSink();

  ui->cancelButton->setEnabled(false);

  // We must have a valid config at this point, initialize the UI with the data from it.
  ui->addressLineEdit->setText(QString::fromStdString(config_.proto().ip_address()));
  ui->portLineEdit->setText(QString::number(config_.proto().port()));
  const bool shouldAutomaticallyConnect = config_.proto().automatically_connect();
  ui->automaticallyConnectCheckBox->setChecked(shouldAutomaticallyConnect);

  // Connect the UI controls.
  connect(ui->connectButton, &QPushButton::clicked, this, &HyperbotConnect::onConnectClicked);
  connect(ui->cancelButton, &QPushButton::clicked, this, &HyperbotConnect::onCancelClicked);
  connect(ui->automaticallyConnectCheckBox, &QCheckBox::clicked, this, &HyperbotConnect::onAutoConnectCheckBoxClicked);

  // Connect Hyperbot's connection signals.
  connect(&hyperbot_, &Hyperbot::connected, this, &HyperbotConnect::handleConnected);
  connect(&hyperbot_, &Hyperbot::connectionFailed, this, &HyperbotConnect::handleConnectionFailed);
  connect(&hyperbot_, &Hyperbot::connectionCancelled, this, &HyperbotConnect::handleConnectionCancelled);

  if (shouldAutomaticallyConnect) {
    tryConnect();
  }
}

HyperbotConnect::~HyperbotConnect() {
  if (myLogSink_ != nullptr) {
    absl::RemoveLogSink(myLogSink_);
    delete myLogSink_;
  }
  delete ui;
}

void HyperbotConnect::onConnectClicked() {
  VLOG(1) << "Connect button clicked";
  tryConnect();
}

void HyperbotConnect::onCancelClicked() {
  VLOG(1) << "Cancel button clicked";
  ui->cancelButton->setEnabled(false);
  hyperbot_.cancelConnect();
}

void HyperbotConnect::onAutoConnectCheckBoxClicked(bool checked) {
  config_.proto().set_automatically_connect(checked);
  config_.save();
}

void HyperbotConnect::handleConnectionFailed() {
  LOG(INFO) << "Failed to connect to Hyperbot.";
  ui->connectButton->setEnabled(true);
  ui->cancelButton->setEnabled(false);
}

void HyperbotConnect::handleConnectionCancelled() {
  LOG(INFO) << "Cancelled connection to Hyperbot.";
  ui->connectButton->setEnabled(true);
  ui->cancelButton->setEnabled(false);
}

void HyperbotConnect::handleConnected() {
  LOG(INFO) << "Connected to Hyperbot.";
  close();
}

void HyperbotConnect::registerLogSink() {
  if (myLogSink_ != nullptr) {
    throw std::runtime_error("Log sink already registered");
  }
  myLogSink_ = new internal::MyLogSink(ui->logTextEdit);
  absl::AddLogSink(myLogSink_);
}

void HyperbotConnect::tryConnect() {
  std::string address = ui->addressLineEdit->text().toStdString();
  int32_t port = ui->portLineEdit->text().toInt();
  config_.proto().set_ip_address(address);
  config_.proto().set_port(port);
  config_.save();

  // Try to connect to the bot.
  ui->cancelButton->setEnabled(true);
  ui->connectButton->setEnabled(false);
  hyperbot_.tryConnectAsync(address, port);
}