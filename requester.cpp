#include "requester.hpp"

Requester::Requester(zmq::context_t &context) : context_(context) {}

void Requester::connect() {
  constexpr const int kDataSize = 5;
  socket_ = zmq::socket_t(context_, zmq::socket_type::req);
  socket_.connect("tcp://localhost:5555");
}

void Requester::sendConfig(const proto::config::Config &config) {
  proto::request::RequestMessage requestMessage;
  *requestMessage.mutable_config() = config;
  serializeSendAndRecvAck(requestMessage);
}

void Requester::injectPacket(proto::request::PacketToInject::Direction packetDirection, uint16_t opcode, const std::string &rawBytes) {
  proto::request::PacketToInject packet;
  packet.set_direction(packetDirection);
  packet.set_opcode(opcode);
  packet.set_data(rawBytes);
  proto::request::RequestMessage requestMessage;
  *requestMessage.mutable_packetdata() = packet;
  serializeSendAndRecvAck(requestMessage);
}

void Requester::setCurrentPositionAsTrainingCenter() {
  proto::request::RequestMessage requestMessage;
  requestMessage.mutable_setcurrentpositionastrainingcenter();
  serializeSendAndRecvAck(requestMessage);
}

void Requester::startTraining() {
  proto::request::DoAction action;
  action.set_action(proto::request::DoAction::kStartTraining);
  proto::request::RequestMessage requestMessage;
  *requestMessage.mutable_doaction() = action;
  serializeSendAndRecvAck(requestMessage);
}

void Requester::stopTraining() {
  proto::request::DoAction action;
  action.set_action(proto::request::DoAction::kStopTraining);
  proto::request::RequestMessage requestMessage;
  *requestMessage.mutable_doaction() = action;
  serializeSendAndRecvAck(requestMessage);
}
