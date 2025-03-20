#ifndef HYPERBOT_HPP_
#define HYPERBOT_HPP_

#include <ui_proto/rl_ui_request.pb.h>

#include <zmq.hpp>

#include <QObject>
#include <QString>

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
  void requestCheckpointList();

signals:
  void connected();
  void connectionFailed();
  void connectionCancelled();
  void checkpointListReceived(const QString &str);

private:
  zmq::context_t context_;
  zmq::socket_t socket_;
  std::thread connectionThread_;
  std::atomic<bool> tryToConnect_;

  void tryConnect();
  bool sendMessage(const proto::rl_ui_request::RequestMessage &message);
};

#endif // HYPERBOT_HPP_
