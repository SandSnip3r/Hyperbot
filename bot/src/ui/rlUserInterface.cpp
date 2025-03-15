#include "broker/eventBroker.hpp"
#include "event/event.hpp"
#include "ui/rlUserInterface.hpp"

#include <ui_proto/rl_ui_request.pb.h>

#include <absl/log/log.h>

namespace ui {

RlUserInterface::RlUserInterface(zmq::context_t &context, broker::EventBroker &eventBroker) : context_(context), eventBroker_(eventBroker) {

}

RlUserInterface::~RlUserInterface() {
  VLOG(1) << "Destructing UserInterface";
  if (thr_.joinable()) {
    keepRunning_ = false;
    thr_.join();
  }
}

void RlUserInterface::initialize() {

}

void RlUserInterface::runAsync() {
  if (thr_.joinable()) {
    throw std::runtime_error("UserInterface::runAsync called while already running");
  }
  // Set up publisher
  try {
    constexpr std::string_view kPublisherAddress{"tcp://*:5556"};
    VLOG(1) << "RlUserInterface:publisher binding to " << kPublisherAddress;
    publisher_.bind(std::string(kPublisherAddress));

    // Run the request receiver in another thread
    keepRunning_ = true;
    thr_ = std::thread(&RlUserInterface::run, this);
  } catch (const std::exception &ex) {
    LOG(ERROR) << "Exception while binding to UI: \"" << ex.what() << "\"";
  } catch (...) {
    LOG(ERROR) << "Exception while binding to UI";
  }
}

void RlUserInterface::run() {
  // Run request receiver
  zmq::socket_t socket(context_, zmq::socket_type::rep);
  constexpr std::string_view kSocketAddress{"tcp://*:5555"};
  VLOG(1) << "RlUserInterface:socket binding to " << kSocketAddress;
  socket.bind(std::string(kSocketAddress));
  while (keepRunning_) {
    // Wait for a request
    zmq::message_t request;
    zmq::pollitem_t items[] = {{socket, 0, ZMQ_POLLIN, 0}};
    std::chrono::duration timeout = std::chrono::milliseconds{100};
    zmq::poll(&items[0], 1, timeout);
    if ((items[0].revents & ZMQ_POLLIN) == 0) {
      // Did not receive anything yet.
      continue;
    }
    zmq::recv_result_t receiveResult = socket.recv(request, zmq::recv_flags::none);
    if (!receiveResult) {
      LOG(WARNING) << "Error receiving message";
      continue;
    }

    handleRequest(request);

    // Immediately respond with an acknowledgement
    const std::string response{"ack"};
    socket.send(zmq::message_t(response), zmq::send_flags::none);
  }
}

void RlUserInterface::handleRequest(const zmq::message_t &request) {
  using namespace proto;
  // Parse the request
  rl_ui_request::RequestMessage requestMsg;
  requestMsg.ParseFromArray(request.data(), request.size());
  LOG(INFO) << "Received request " << requestMsg.DebugString();
  switch (requestMsg.body_case()) {
    case rl_ui_request::RequestMessage::BodyCase::kDoAction: {
      const rl_ui_request::DoAction &doActionMsg = requestMsg.doaction();
      if (doActionMsg.action() == rl_ui_request::DoAction::kStartTraining) {
        eventBroker_.publishEvent(event::EventCode::kStarRlTraining);
      }
      break;
    }
    case rl_ui_request::RequestMessage::BodyCase::kPing: {
      LOG(INFO) << "Received ping";
      break;
    }
    default:
    LOG(WARNING) << "Unknown request type";
    break;
  }
}

} // namespace ui