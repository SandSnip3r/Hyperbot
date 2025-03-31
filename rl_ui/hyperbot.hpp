#ifndef HYPERBOT_HPP_
#define HYPERBOT_HPP_

#include "hyperbotConnectWorker.hpp"
#include "hyperbotSubscriberWorker.hpp"

#include <ui_proto/rl_ui_messages.pb.h>

#include <zmq.hpp>

#include <QObject>
#include <QStringList>
#include <QTimer>
#include <QThread>

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>

class Hyperbot : public QObject {
  Q_OBJECT
public:
  ~Hyperbot();
  // Tries to connect to the Hyperbot server. Returns true if successful.
  void tryConnectAsync(std::string_view ipAddress, int32_t port);
  void cancelConnect();

  void startTraining();
  void stopTraining();
  void requestCheckpointList();
  void saveCheckpoint(const QString &checkpointName);
  void loadCheckpoint(const QString &checkpointName);

public slots:
  void onConnectionFailed();
  void onConnectionCancelled();
  void onConnected(int broadcastPort);
  void handleBroadcastMessage(proto::rl_ui_messages::BroadcastMessage broadcastMessage);
  void onSubscriberDisconnected();

signals:
  void connectionFailed();
  void connectionCancelled();
  void connected();
  void disconnected();

  // Broadcast messages.
  void checkpointListReceived(QStringList str);
  void checkpointAlreadyExists(QString checkpointName);
  void savingCheckpoint();
  void plotData(qreal x, qreal y);

private:
  static constexpr int kHeartbeatIntervalMs = 500;
  zmq::context_t context_;
  std::string ipAddress_;
  std::atomic<bool> connected_;
  zmq::socket_t socket_;
  QThread *connectThread_{nullptr};
  QThread *subscriberThread_{nullptr};
  HyperbotConnectWorker *connectWorker_{nullptr};
  HyperbotSubscriberWorker *subscriberWorker_{nullptr};


  // void tryConnect();
  void setupSubscriber(int broadcastPort);
  void sendAsyncRequest(const proto::rl_ui_messages::AsyncRequest &asyncRequest);
  bool sendMessage(const proto::rl_ui_messages::RequestMessage &message);
  void subscriberThreadFunc();
};

#endif // HYPERBOT_HPP_
