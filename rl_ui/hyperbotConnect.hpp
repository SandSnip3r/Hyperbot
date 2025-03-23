#ifndef HYPERBOT_CONNECT_HPP_
#define HYPERBOT_CONNECT_HPP_

#include "config.hpp"
#include "hyperbot.hpp"

#include <absl/log/log.h>
#include <absl/log/log_entry.h>
#include <absl/log/log_sink.h>

#include <QMainWindow>
#include <QTextEdit>

namespace Ui {
class HyperbotConnect;
}

namespace internal {

class MyLogSink : public absl::LogSink {
public:
  MyLogSink(QTextEdit *loggingTextEdit) : loggingTextEdit_(loggingTextEdit) {}
  void Send(const absl::LogEntry& entry) override;
private:
  QTextEdit *loggingTextEdit_;
};

} // namespace internal

class HyperbotConnect : public QMainWindow {
  Q_OBJECT

public:
  explicit HyperbotConnect(Config &config, Hyperbot &hyperbot, QWidget *parent = nullptr);
  ~HyperbotConnect();

signals:
private slots:
  void onConnectClicked();
  void onCancelClicked();
  void onAutoConnectCheckBoxClicked(bool checked);
  void handleConnectionFailed();
  void handleConnectionCancelled();
  void handleConnected();
private:
  Ui::HyperbotConnect *ui;
  Config &config_;
  Hyperbot &hyperbot_;
  internal::MyLogSink *myLogSink_{nullptr};

  void registerLogSink();
  void tryConnect();
};

#endif // HYPERBOT_CONNECT_HPP_
