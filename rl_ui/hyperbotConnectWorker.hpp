#ifndef HYPERBOT_CONNECT_WORKER_HPP_
#define HYPERBOT_CONNECT_WORKER_HPP_

#include <zmq.hpp>

#include <QObject>

#include <atomic>
#include <string>

// Worker for connection logic.
class HyperbotConnectWorker : public QObject {
  Q_OBJECT
public:
  HyperbotConnectWorker(const std::string &ipAddress, int port, zmq::socket_t &socket, QObject *parent=nullptr);
  ~HyperbotConnectWorker();

  void stopTrying() { tryToConnect_ = false; }

public slots:
  void process();

signals:
  void connected(int broadcastPort);
  void connectionFailed();
  void connectionCancelled();

private:
  const std::string ipAddress_;
  const int port_;
  zmq::socket_t &socket_;
  std::atomic<bool> tryToConnect_{true};
};

#endif // HYPERBOT_CONNECT_WORKER_HPP_
