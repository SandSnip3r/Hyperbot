#include "requester.hpp"

Requester::Requester(zmq::context_t &context) : context_(context) {}

void Requester::connect() {
  constexpr const int kDataSize = 5;
  socket_ = zmq::socket_t(context_, zmq::socket_type::req);
  socket_.connect("tcp://localhost:5555");
}

void Requester::injectPacket(request::PacketToInject::Direction packetDirection, uint16_t opcode, const std::string &rawBytes) {
  request::PacketToInject packet;
  packet.set_direction(packetDirection);
  packet.set_opcode(opcode);
  packet.set_data(rawBytes);
  request::RequestMessage requestMessage;
  *requestMessage.mutable_packetdata() = packet;
  serializeSendAndRecvAck(requestMessage);
}

void Requester::startTraining() {
  request::DoAction action;
  action.set_action(request::DoAction::kStartTraining);
  request::RequestMessage requestMessage;
  *requestMessage.mutable_doaction() = action;
  serializeSendAndRecvAck(requestMessage);
}

void Requester::stopTraining() {
  request::DoAction action;
  action.set_action(request::DoAction::kStopTraining);
  request::RequestMessage requestMessage;
  *requestMessage.mutable_doaction() = action;
  serializeSendAndRecvAck(requestMessage);
}
