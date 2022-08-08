#include "eventHandler.hpp"

EventHandler::EventHandler(zmq::context_t &context) : context_(context) {}

EventHandler::~EventHandler() {
  run_ = false;
  if (thr_.joinable()) {
    thr_.join();
  }
}

void EventHandler::runAsync() {
  run_ = true;
  thr_ = std::thread(&EventHandler::run, this);
}

void EventHandler::run() {
  // Set up subscriber socket
  zmq::socket_t subscriber(context_, zmq::socket_type::sub);
  subscriber.set(zmq::sockopt::subscribe, "");
  subscriber.connect("tcp://localhost:5556");
  emit connected();
  while (run_) {
    zmq::message_t message;
    subscriber.recv(message);
    broadcast::BroadcastMessage broadcastMessage;
    broadcastMessage.ParseFromArray(message.data(), message.size());
    handle(broadcastMessage);
  }
}

void EventHandler::handle(const broadcast::BroadcastMessage &message) {
  switch (message.body_case()) {
    case broadcast::BroadcastMessage::BodyCase::kMessage1: {
        // std::lock_guard<std::mutex> lk(printMutex);
        // std::cout << "Received an instance of Message1" << std::endl;
        const broadcast::Message1 &message1 = message.message1();
        // std::cout << "  Data: " << message1.data() << std::endl;
        emit message1Received(message1.data());
        break;
      }
    case broadcast::BroadcastMessage::BodyCase::kMessage2: {
        // std::lock_guard<std::mutex> lk(printMutex);
        // std::cout << "Received an instance of Message2" << std::endl;
        const broadcast::Message2 &message2 = message.message2();
        // std::cout << "  Data: " << message2.data() << std::endl;
        emit message2Received(message2.data());
        break;
      }
    case broadcast::BroadcastMessage::BodyCase::kMessage3: {
        // std::lock_guard<std::mutex> lk(printMutex);
        // std::cout << "Received an instance of Message3" << std::endl;
        const broadcast::Message3 &message3 = message.message3();
        // std::cout << "  Data: " << message3.data() << std::endl;
        emit message3Received(message3.data());
        break;
      }
    case broadcast::BroadcastMessage::BodyCase::kHpMpUpdate: {
        const broadcast::HpMpUpdate &hpMpUpdate = message.hpmpupdate();
        emit vitalsChanged(hpMpUpdate);
        break;
      }
  }
}