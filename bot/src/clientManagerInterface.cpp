#include "clientManagerInterface.hpp"

#include <ui_proto/client_manager_request.pb.h>

#include <absl/log/log.h>
#include <absl/strings/str_format.h>

#include <stdexcept>

ClientManagerInterface::ClientManagerInterface(zmq::context_t &context) : context_(context) {
  socket_ = zmq::socket_t(context_, zmq::socket_type::req);
  socket_.bind("tcp://*:2235");

  // // When this process is killed, send requests to kill any open clients.
  // signal(SIGINT, &ClientManagerInterface::signalHandler);
}

ClientManagerInterface::~ClientManagerInterface() {
  std::unique_lock lock(mutex_);
  VLOG(1) << "Destructing";
  shouldStop_ = true;
  conditionVariable_.notify_all();
  if (runThread_.joinable()) {
    runThread_.join();
  } else {
    LOG(WARNING) << "Weird, thread wasn't joinable";
  }
}

void ClientManagerInterface::runAsync() {
  if (running_) {
    throw std::runtime_error("ClientManagerInterface: Already running");
  }
  runThread_ = std::thread(&ClientManagerInterface::run, this);
}

ClientManagerInterface::ClientId ClientManagerInterface::startClient(int32_t listeningPort) {
  if (!running_) {
    throw std::runtime_error("ClientManagerInterface: Not running");
  }
  VLOG(1) << "Asked to start a client on port " << listeningPort;
  std::future<ClientId> future;
  {
    std::unique_lock lock(mutex_);
    VLOG(1) << "Pushing port " << listeningPort << " onto the list of ports pending open, also creating promise/future";
    {
      ClientOpenRequest request;
      request.port = listeningPort;
      future = request.completedPromise.get_future();
      clientsPendingOpen_.push_back(std::move(request));
    }
    conditionVariable_.notify_all();
  }
  VLOG(1) << "Waiting on future";
  const ClientId id = future.get();
  VLOG(1) << "Promise fulfilled, returning client id " << id;
  return id;
}

ClientManagerInterface::ClientId ClientManagerInterface::privateStartClient(int32_t listeningPort) {
  VLOG(1) << "Sending request to start client, which should connect to port " << listeningPort;
  sendClientStartRequest(listeningPort);

  VLOG(2) << "Message sent, now awaiting reply";
  zmq::message_t reply;
  std::optional<zmq::recv_result_t> receiveResult = socket_.recv(reply, zmq::recv_flags::none);
  if (!receiveResult) {
    throw std::runtime_error("ClientManagerInterface: Failed to receive reply while trying to start sro_client");
  }

  proto::client_manager_request::Response response;
  bool successfullyParsed = response.ParseFromArray(reply.data(), reply.size());
  if (!successfullyParsed) {
    throw std::runtime_error("ClientManagerInterface: Failed to parse response while trying to start sro_client");
  }

  VLOG(3) << "Received reply:\n" << response.DebugString();
  switch (response.body_case()) {
    case proto::client_manager_request::Response::BodyCase::kClientStarted: {
      const int32_t clientId = response.client_started().client_id();
      // saveClientId(clientId);
      return clientId;
    }
    case proto::client_manager_request::Response::BodyCase::kError: {
      throw std::runtime_error(absl::StrFormat("ClientManagerInterface: Error received: \"%s\"", response.error().message()));
    }
    default: {
      throw std::runtime_error(absl::StrFormat("ClientManagerInterface: Unexpected response body case %d", static_cast<int>(response.body_case())));
    }
  }
}

void ClientManagerInterface::sendClientStartRequest(int32_t port) {
  proto::client_manager_request::StartClient startClient;
  startClient.set_port_to_connect_to(port);
  proto::client_manager_request::Request request;
  request.mutable_start_client()->CopyFrom(startClient);

  VLOG(3) << "Sending:\n" << request.DebugString();

  std::optional<zmq::send_result_t> sendResult = socket_.send(zmq::message_t(request.SerializeAsString()), zmq::send_flags::none);
  if (!sendResult) {
    throw std::runtime_error("ClientManagerInterface: Failed to send message to start sro_client");
  }
}

void ClientManagerInterface::sendHeartbeat() {
  proto::client_manager_request::Request request;
  request.mutable_heartbeat();
  VLOG(10) << "Sending:\n" << request.DebugString();
  std::optional<zmq::send_result_t> sendResult = socket_.send(zmq::message_t(request.SerializeAsString()), zmq::send_flags::none);
  if (!sendResult) {
    throw std::runtime_error("ClientManagerInterface: Failed to send heartbeat");
  }

  VLOG(10) << "Message sent, now awaiting reply";
  zmq::message_t reply;
  std::optional<zmq::recv_result_t> receiveResult = socket_.recv(reply, zmq::recv_flags::none);
  if (!receiveResult) {
    throw std::runtime_error("ClientManagerInterface: Failed to receive reply to heartbeat");
  }

  proto::client_manager_request::Response response;
  bool successfullyParsed = response.ParseFromArray(reply.data(), reply.size());
  if (!successfullyParsed) {
    throw std::runtime_error("ClientManagerInterface: Failed to parse response to heartbeat");
  }

  VLOG(10) << "Received reply:\n" << response.DebugString();
  switch (response.body_case()) {
    case proto::client_manager_request::Response::BodyCase::kHeartbeatAck: {
      VLOG(10) << "Heartbeat acknowledged";
      break;
    }
    default: {
      throw std::runtime_error(absl::StrFormat("ClientManagerInterface: Unexpected response body case %d", static_cast<int>(response.body_case())));
    }
  }
}

void ClientManagerInterface::run() {
  running_ = true;
  VLOG(1) << "Running";
  while (true) {
    // Wait on our condition variable
    std::unique_lock lock(mutex_);
    conditionVariable_.wait_for(lock, kMaxHeartbeatSilence, [this]() -> bool {
      // Should we stop waiting?
      const std::chrono::time_point<std::chrono::high_resolution_clock> now = std::chrono::high_resolution_clock::now();
      if (shouldStop_) {
        // We've been told to stop running.
        return true;
      }
      if (!clientsPendingOpen_.empty()) {
        // There is a request to open a client.
        return true;
      }
      if (now - lastMessageSent_ > kMaxHeartbeatSilence) {
        // It has been too long since we last sent a message.
        return true;
      }
      // Keep waiting.
      return false;
    });
    // We've been woken up! Why?
    VLOG(10) << "We're awake!";

    // Highest priority is to quit if we've been told to.
    if (shouldStop_) {
      VLOG(1) << "Quitting";
      return;
    }

    // Next priority is to open a client if we've been asked to do so.
    if (!clientsPendingOpen_.empty()) {
      ClientOpenRequest request = std::move(clientsPendingOpen_.front());
      clientsPendingOpen_.pop_front();
      VLOG(1) << "Sending message to open a client listening on port " << request.port;
      const ClientId id = privateStartClient(request.port);
      VLOG(1) << "Client opened with ID " << id << ". Fulfilling promise";
      request.completedPromise.set_value(id);

      // We've sent a message. Update our last message sent time.
      lastMessageSent_ = std::chrono::high_resolution_clock::now();
      continue;
    }

    // We should not stop and there are no client open requests. We've been woken up because it has been too long since we've sent a message. Send a heartbeat.
    const auto diffMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - lastMessageSent_);
    VLOG(10) << diffMs.count() << "ms () since last message sent. Sending heartbeat";
    sendHeartbeat();
    lastMessageSent_ = std::chrono::high_resolution_clock::now();
  }
}

// void ClientManagerInterface::handleReply(zmq::message_t reply) {
// }

// void ClientManagerInterface::saveClientId(ClientId clientId) {
//   // std::unique_lock<std::mutex> lock(runningClientListMutex_);
//   LOG(INFO) << "Saving client " << clientId << " to be killed later";
//   runningClients_.push_back(clientId);
// }

// std::vector<ClientManagerInterface::ClientId> ClientManagerInterface::runningClients_;

// void ClientManagerInterface::signalHandler(int signal) {
//   // std::unique_lock<std::mutex> lock(runningClientListMutex_);
//   for (ClientId clientId : runningClients_) {
//     LOG(INFO) << "Killing client " << clientId;
//   }
//   exit(0);
// }