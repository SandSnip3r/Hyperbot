#ifndef HYPERBOT_SUBSCRIBER_WORKER_HPP_
#define HYPERBOT_SUBSCRIBER_WORKER_HPP_

#include <ui_proto/rl_ui_messages.pb.h>

#include <zmq.hpp>

#include <QElapsedTimer>
#include <QObject>

// Worker for subscriber logic and heartbeat detection without QTimer.
class HyperbotSubscriberWorker : public QObject {
  Q_OBJECT
public:
  HyperbotSubscriberWorker(zmq::context_t &context, std::string ipAddress, int port, QObject *parent = nullptr);
  ~HyperbotSubscriberWorker();

public slots:
  // This slot will run in a tight loop.
  void startWork();

signals:
  void disconnected();
  void broadcastMessageReceived(proto::rl_ui_messages::BroadcastMessage broadcastMessage);

private:
  void processMessage(const zmq::message_t &message, QElapsedTimer &lastHeartbeatTimer);

  zmq::socket_t subscriber_;
};

#endif // HYPERBOT_SUBSCRIBER_WORKER_HPP_
