#ifndef REQUESTER_HPP_
#define REQUESTER_HPP_

#include "ui-proto/old_config.pb.h"
#include "ui-proto/request.pb.h"

#include <zmq.hpp>

#include <QObject>

#include <string>

class Requester : public QObject {
  Q_OBJECT
public:
  Requester(zmq::context_t &context);
  void connect();
  void sendConfig(const proto::old_config::Config &config);
  void injectPacket(proto::request::PacketToInject::Direction packetDirection, uint16_t opcode, const std::string &rawBytes);
  void setCurrentPositionAsTrainingCenter();
  void startTraining();
  void stopTraining();
private:
  zmq::context_t &context_;
  zmq::socket_t socket_;

  template<typename T>
  void serializeSendAndRecvAck(const T &protoMsg) {
    std::string protoMsgAsStr;
    protoMsg.SerializeToString(&protoMsgAsStr);

    zmq::message_t zmqMsg;
    zmqMsg.rebuild(protoMsgAsStr.data(), protoMsgAsStr.size());
    socket_.send(zmqMsg, zmq::send_flags::none);

    // Receive our quick acknowledgement
    zmq::message_t reply;
    socket_.recv(reply, zmq::recv_flags::none);
  }
signals:
};

#endif // REQUESTER_HPP_
