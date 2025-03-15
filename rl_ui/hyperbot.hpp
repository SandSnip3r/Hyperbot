#ifndef HYPERBOT_HPP_
#define HYPERBOT_HPP_

#include <zmq.hpp>

#include <QObject>

#include <atomic>
#include <cstdint>
#include <string_view>
#include <thread>

class Hyperbot : public QObject {
  Q_OBJECT
public:
  ~Hyperbot();
  // Tries to connect to the Hyperbot server. Returns true if successful.
  void tryConnectAsync(std::string_view ipAddress, int32_t port);
  void cancelConnect();

  void startTraining();

signals:
  void connected();
  void connectionFailed();
  void connectionCancelled();

private:
  zmq::context_t context_;
  zmq::socket_t socket_;
  std::thread connectionThread_;
  std::atomic<bool> tryToConnect_;

  void tryConnect();
};

#endif // HYPERBOT_HPP_
