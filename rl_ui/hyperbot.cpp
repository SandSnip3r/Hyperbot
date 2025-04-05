#include "hyperbot.hpp"

#include <absl/log/log.h>
#include <absl/strings/str_format.h>

using namespace proto;

Hyperbot::~Hyperbot() {
  if (connectThread_ && connectThread_->isRunning()) {
    connectThread_->quit();
    connectThread_->wait();
  }
  if (subscriberThread_ && subscriberThread_->isRunning()) {
    subscriberThread_->requestInterruption();
    subscriberThread_->quit();
    subscriberThread_->wait();
  }
}

void Hyperbot::tryConnectAsync(std::string_view ipAddress, int32_t port) {
  connected_ = false;
  ipAddress_ = ipAddress;
  LOG(INFO) << absl::StrFormat("Connecting to Hyperbot at %s:%d.", ipAddress_, port);
  socket_ = zmq::socket_t(context_, zmq::socket_type::req);
  socket_.connect(absl::StrFormat("tcp://%s:%d", ipAddress_, port));

  // Set up the connection worker in its own thread.
  connectThread_ = new QThread;
  connectWorker_ = new HyperbotConnectWorker(ipAddress_, port, socket_);
  connectWorker_->moveToThread(connectThread_);

  connect(connectThread_, &QThread::started, connectWorker_, &HyperbotConnectWorker::process);
  connect(connectWorker_, &HyperbotConnectWorker::connectionFailed, this, &Hyperbot::onConnectionFailed);
  connect(connectWorker_, &HyperbotConnectWorker::connectionCancelled, this, &Hyperbot::onConnectionCancelled);
  connect(connectWorker_, &HyperbotConnectWorker::connected, this, &Hyperbot::onConnected);

  // Ensure thread cleanup.
  connect(connectWorker_, &HyperbotConnectWorker::connected, connectThread_, &QThread::quit);
  connect(connectWorker_, &HyperbotConnectWorker::connectionFailed, connectThread_, &QThread::quit);
  connect(connectWorker_, &HyperbotConnectWorker::connectionCancelled, connectThread_, &QThread::quit);
  connect(connectThread_, &QThread::finished, [this]() {
    delete connectWorker_;
    connectWorker_ = nullptr;
    connectThread_->deleteLater();
    connectThread_ = nullptr;
  });

  connectThread_->start();
}

void Hyperbot::cancelConnect() {
  if (connectWorker_) {
    connectWorker_->stopTrying();
  }
}

void Hyperbot::startTraining() {
  rl_ui_messages::AsyncRequest asyncRequest;
  asyncRequest.mutable_start_training();
  sendAsyncRequest(asyncRequest);
}

void Hyperbot::stopTraining() {
  rl_ui_messages::AsyncRequest asyncRequest;
  asyncRequest.mutable_stop_training();
  sendAsyncRequest(asyncRequest);
}

void Hyperbot::requestCheckpointList() {
  rl_ui_messages::AsyncRequest asyncRequest;
  asyncRequest.mutable_request_checkpoint_list();
  sendAsyncRequest(asyncRequest);
}

void Hyperbot::saveCheckpoint(const QString &checkpointName) {
  rl_ui_messages::AsyncRequest asyncRequest;
  rl_ui_messages::SaveCheckpoint *saveCheckpoint = asyncRequest.mutable_save_checkpoint();
  saveCheckpoint->set_name(checkpointName.toStdString());
  sendAsyncRequest(asyncRequest);
}

void Hyperbot::loadCheckpoint(const QString &checkpointName) {
  rl_ui_messages::AsyncRequest asyncRequest;
  rl_ui_messages::LoadCheckpoint *loadCheckpoint = asyncRequest.mutable_load_checkpoint();
  loadCheckpoint->set_name(checkpointName.toStdString());
  sendAsyncRequest(asyncRequest);
}

void Hyperbot::deleteCheckpoints(const QList<QString> &checkpointNames) {
  rl_ui_messages::AsyncRequest asyncRequest;
  rl_ui_messages::DeleteCheckpoints *deleteCheckpoints = asyncRequest.mutable_delete_checkpoints();
  for (const QString &checkpointName : checkpointNames) {
    deleteCheckpoints->add_name(checkpointName.toStdString());
  }
  sendAsyncRequest(asyncRequest);
}

void Hyperbot::onConnectionFailed() {
  emit connectionFailed();
}

void Hyperbot::onConnectionCancelled() {
 emit connectionCancelled();
}

void Hyperbot::onConnected(int broadcastPort) {
  connected_ = true;
  setupSubscriber(broadcastPort);
  emit connected();
}

void Hyperbot::handleBroadcastMessage(proto::rl_ui_messages::BroadcastMessage broadcastMessage) {
  switch (broadcastMessage.body_case()) {
    case rl_ui_messages::BroadcastMessage::BodyCase::kHeartbeat: {
      LOG(WARNING) << "Should not receive heartbeat message here.";
      break;
    }
    case rl_ui_messages::BroadcastMessage::BodyCase::kCheckpointList: {
      const rl_ui_messages::CheckpointList &checkpointList = broadcastMessage.checkpoint_list();
      QStringList checkpointListStr;
      for (const rl_ui_messages::Checkpoint &checkpoint : checkpointList.checkpoints()) {
        checkpointListStr.append(QString::fromStdString(checkpoint.name()));
      }
      emit checkpointListReceived(checkpointListStr);
      break;
    }
    case rl_ui_messages::BroadcastMessage::BodyCase::kCheckpointAlreadyExists: {
      emit checkpointAlreadyExists(QString::fromStdString(broadcastMessage.checkpoint_already_exists()));
      break;
    }
    case rl_ui_messages::BroadcastMessage::BodyCase::kSavingCheckpoint: {
      emit savingCheckpoint();
      break;
    }
    case rl_ui_messages::BroadcastMessage::BodyCase::kPlotData: {
      const rl_ui_messages::PlotData &plotDataMsg = broadcastMessage.plot_data();
      emit plotData(plotDataMsg.x(), plotDataMsg.y());
      break;
    }
    default: {
      LOG(WARNING) << "Received unexpected broadcast message.";
      break;
    }
  }
}

void Hyperbot::onSubscriberDisconnected() {
  emit disconnected();
}

void Hyperbot::setupSubscriber(int broadcastPort) {
  // Set up the subscriber worker in its own thread.
  subscriberThread_ = new QThread;
  subscriberWorker_ = new HyperbotSubscriberWorker(context_, ipAddress_, broadcastPort);
  subscriberWorker_->moveToThread(subscriberThread_);

  connect(subscriberThread_, &QThread::started, subscriberWorker_, &HyperbotSubscriberWorker::startWork);
  connect(subscriberWorker_, &HyperbotSubscriberWorker::disconnected, this, &Hyperbot::onSubscriberDisconnected);
  connect(subscriberWorker_, &HyperbotSubscriberWorker::broadcastMessageReceived, this, &Hyperbot::handleBroadcastMessage);

  // Clean up when the subscriber worker signals a disconnect.
  connect(subscriberWorker_, &HyperbotSubscriberWorker::disconnected, subscriberThread_, &QThread::quit);
  connect(subscriberThread_, &QThread::finished, [this]() {
    delete subscriberWorker_;
    subscriberWorker_ = nullptr;
    subscriberThread_->deleteLater();
    subscriberThread_ = nullptr;
  });

  subscriberThread_->start();
}

void Hyperbot::sendAsyncRequest(const rl_ui_messages::AsyncRequest &asyncRequest) {
  if (!connected_) {
    LOG(WARNING) << "Not connected to Hyperbot.";
    return;
  }
  rl_ui_messages::RequestMessage asyncRequestMessage;
  *asyncRequestMessage.mutable_async_request() = asyncRequest;
  std::string protoMsgAsStr;
  asyncRequestMessage.SerializeToString(&protoMsgAsStr);
  zmq::message_t zmqMsg(protoMsgAsStr);
  zmq::send_result_t sendResult = socket_.send(zmqMsg, zmq::send_flags::none);
  if (!sendResult) {
    LOG(WARNING) << "Failed to send message to Hyperbot.";
    return;
  }
  // TODO: Poll for reply. If polling fails, Hyperbot might've died immediately after our send. We should reset the socket and exit.
  zmq::message_t reply;
  zmq::recv_result_t receiveResult = socket_.recv(reply, zmq::recv_flags::none);
  if (!receiveResult) {
    LOG(WARNING) << "Failed to receive reply";
    return;
  }

  rl_ui_messages::ReplyMessage replyMsg;
  bool parseSuccess = replyMsg.ParseFromArray(reply.data(), reply.size());
  if (!parseSuccess) {
    LOG(WARNING) << "Failed to parse reply";
    return;
  }
  VLOG(1) << "Successfully parsed reply: " << replyMsg.DebugString();
  if (replyMsg.body_case() != rl_ui_messages::ReplyMessage::BodyCase::kAsyncRequestAck) {
    LOG(WARNING) << "Received unexpected reply";
    return;
  }
}

bool Hyperbot::sendMessage(const rl_ui_messages::RequestMessage &message) {
  std::string protoMsgAsStr;
  message.SerializeToString(&protoMsgAsStr);
  zmq::message_t zmqMsg(protoMsgAsStr);
  std::optional<zmq::send_result_t> sendResult = socket_.send(zmqMsg, zmq::send_flags::none);

  if (!sendResult.has_value()) {
    LOG(WARNING) << "Failed to send message to Hyperbot.";
    return false;
  }
  return true;
}