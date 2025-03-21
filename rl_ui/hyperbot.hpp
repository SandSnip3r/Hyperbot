#ifndef HYPERBOT_HPP_
#define HYPERBOT_HPP_

#include <ui_proto/rl_ui_messages.pb.h>

#include <zmq.hpp>

#include <QObject>
#include <QStringList>

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

signals:
  void connected();
  void connectionFailed();
  void connectionCancelled();
  void checkpointListReceived(QStringList str);

private:
  zmq::context_t context_;
  zmq::socket_t socket_;
  std::string ipAddress_;
  std::thread connectionThread_;
  std::atomic<bool> tryToConnect_;
  zmq::socket_t subscriber_;
  std::thread subscriberThread_;

  void tryConnect();
  void doAction(proto::rl_ui_messages::DoAction::Action action);
  bool sendMessage(const proto::rl_ui_messages::RequestMessage &message);
  void subscriberThreadFunc();
};

#endif // HYPERBOT_HPP_
