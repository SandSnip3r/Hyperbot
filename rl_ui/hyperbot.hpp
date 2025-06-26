#ifndef HYPERBOT_HPP_
#define HYPERBOT_HPP_

#include "hyperbotConnectWorker.hpp"
#include "hyperbotSubscriberWorker.hpp"

#include <ui_proto/rl_ui_messages.pb.h>
#include <silkroad_lib/scalar_types.hpp>

#include <zmq.hpp>

#include <QObject>
#include <QStringList>
#include <QList>
#include <QTimer>
#include <QThread>
#include <QVector>
#include <QMetaType>
#include "characterDetailDialog.hpp"

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>

struct CheckpointInfo {
  QString name;
  qint64 timestampMs{0};
  int trainStepCount{0};
};

Q_DECLARE_METATYPE(CheckpointInfo)

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
  void requestCharacterStatuses();
  void saveCheckpoint(const QString &checkpointName);
  void loadCheckpoint(const QString &checkpointName);
  void deleteCheckpoints(const QList<QString> &checkpointNames);

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
  void checkpointListReceived(QList<CheckpointInfo> checkpoints);
  void checkpointAlreadyExists(QString checkpointName);
  void savingCheckpoint();
  void checkpointLoaded(QString checkpointName);
  void plotData(qreal x, qreal y);
  void characterStatusReceived(QString name, int currentHp, int maxHp,
                               int currentMp, int maxMp);
  void activeStateMachineReceived(QString name, QString stateMachine);
  void skillCooldownsReceived(QString name, QList<SkillCooldown> cooldowns);
  void qValuesReceived(QString name, QVector<float> qValues);
  void itemCountReceived(QString name,
                         sro::scalar_types::ReferenceObjectId itemRefId,
                         int count);

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
