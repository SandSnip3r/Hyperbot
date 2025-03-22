#include "hyperbotSubscriberWorker.hpp"

#include <QThread>

#include <absl/log/log.h>
#include <absl/strings/str_format.h>

using namespace proto;

HyperbotSubscriberWorker::HyperbotSubscriberWorker(zmq::context_t &context, std::string ipAddress, int port, QObject *parent) : QObject(parent) {
  subscriber_ = zmq::socket_t(context, zmq::socket_type::sub);
  subscriber_.set(zmq::sockopt::subscribe, "");
  subscriber_.connect(absl::StrFormat("tcp://%s:%d", ipAddress, port));
}

HyperbotSubscriberWorker::~HyperbotSubscriberWorker() {
}

void HyperbotSubscriberWorker::startWork() {
  constexpr int kHeartbeatTimeoutMs = 750;
  QElapsedTimer lastHeartbeatTimer;
  lastHeartbeatTimer.start();

  while (!QThread::currentThread()->isInterruptionRequested()) {
    constexpr std::chrono::milliseconds kPollTimeout(100);
    std::vector<zmq::pollitem_t> items = { { subscriber_, 0, ZMQ_POLLIN, 0 } };
    int rc;
    try {
      rc = zmq::poll(items, kPollTimeout);
    } catch (const zmq::error_t &e) {
      LOG(WARNING) << "ZMQ poll error: " << e.what();
      break;
    }
    if (rc > 0 && (items[0].revents & ZMQ_POLLIN)) {
      zmq::message_t message;
      zmq::recv_result_t result = subscriber_.recv(message, zmq::recv_flags::none);
      if (result) {
        processMessage(message, lastHeartbeatTimer);
      }
    }
    if (lastHeartbeatTimer.elapsed() >= kHeartbeatTimeoutMs) {
      LOG(WARNING) << "Did not receive heartbeat from Hyperbot.";
      break;
    }
  }
  emit disconnected();
}

void HyperbotSubscriberWorker::processMessage(const zmq::message_t &message, QElapsedTimer &lastHeartbeatTimer) {
  rl_ui_messages::BroadcastMessage broadcastMessage;
  if (!broadcastMessage.ParseFromArray(message.data(), message.size())) {
    LOG(WARNING) << "Failed to parse broadcast message";
    return;
  }
  if (broadcastMessage.body_case() == rl_ui_messages::BroadcastMessage::BodyCase::kHeartbeat) {
    // A heartbeat was received; restart the timer.
    lastHeartbeatTimer.restart();
    // Return early to avoid emitting the heartbeat message, since nobody else cares about it.
    return;
  }
  emit broadcastMessageReceived(broadcastMessage);
}