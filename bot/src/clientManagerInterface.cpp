#include "broker/eventBroker.hpp"
#include "clientManagerInterface.hpp"

#include <ui_proto/client_manager_request.pb.h>

// Tracy
#include <common/TracySystem.hpp>

#include <absl/log/log.h>
#include <absl/strings/str_format.h>

#include <stdexcept>

ClientManagerInterface::ClientManagerInterface(zmq::context_t &context, broker::EventBroker &eventBroker) : context_(context), eventBroker_(eventBroker) {
  socket_ = zmq::socket_t(context_, zmq::socket_type::req);
  socket_.bind("tcp://*:2235");
}

ClientManagerInterface::~ClientManagerInterface() {
  {
    shouldStop_ = true;
    std::unique_lock lock(mutex_);
    VLOG(1) << "Destructing";
  }
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
  VLOG(20) << "Sending:\n" << request.DebugString();
  std::optional<zmq::send_result_t> sendResult = socket_.send(zmq::message_t(request.SerializeAsString()), zmq::send_flags::none);
  if (!sendResult) {
    throw std::runtime_error("ClientManagerInterface: Failed to send heartbeat");
  }

  VLOG(20) << "Message sent, now awaiting reply";
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

  VLOG(20) << "Received reply:\n" << response.DebugString();
  switch (response.body_case()) {
    case proto::client_manager_request::Response::BodyCase::kHeartbeatAck: {
      VLOG(20) << "Heartbeat acknowledged";
      break;
    }
    case proto::client_manager_request::Response::BodyCase::kClientsDied: {
      // If a client process dies, we'll find out about it in the response to our heartbeat.
      VLOG(1) << response.clients_died().client_ids_size() << " client(s) have died";
      for (int i=0; i<response.clients_died().client_ids_size(); ++i) {
        VLOG(1) << "Publishing event that client " << response.clients_died().client_ids(i) << " has died";
        eventBroker_.publishEvent<event::ClientDied>(response.clients_died().client_ids(i));
      }
      break;
    }
    default: {
      throw std::runtime_error(absl::StrFormat("ClientManagerInterface: Unexpected response body case %d", static_cast<int>(response.body_case())));
    }
  }
}

void ClientManagerInterface::run() {
  tracy::SetThreadName("ClientManagerInterface");
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
    VLOG(20) << "We're awake!";

    // Highest priority is to quit if we've been told to.
    if (shouldStop_) {
      VLOG(1) << "Quitting";
      running_ = false;
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
    VLOG(20) << diffMs.count() << "ms () since last message sent. Sending heartbeat";
    try {
      sendHeartbeat();
      lastMessageSent_ = std::chrono::high_resolution_clock::now();
    } catch (const std::exception &ex) {
      LOG(ERROR) << "Error while running ClientManagerInterface: \"" << ex.what() << "\"";
    }
  }
  running_ = false;
}