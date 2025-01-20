#include "clientManagerInterface.hpp"

#include <ui_proto/client_manager_request.pb.h>

#include <absl/log/log.h>
#include <absl/strings/str_format.h>

#include <stdexcept>

ClientManagerInterface::ClientManagerInterface(zmq::context_t &context) : context_(context) {
  socket_ = zmq::socket_t(context_, zmq::socket_type::req);
  socket_.bind("tcp://*:2235");
}

ClientManagerInterface::ClientId ClientManagerInterface::startClient(int32_t listeningPort) {
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

// void ClientManagerInterface::handleReply(zmq::message_t reply) {
// }
