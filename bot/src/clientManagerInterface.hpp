#ifndef CLIENTMANAGERINTERFACE_HPP_
#define CLIENTMANAGERINTERFACE_HPP_

#include <zmq.hpp>

#include <cstdint>

class ClientManagerInterface {
public:
  using ClientId = int32_t;

  ClientManagerInterface(zmq::context_t &context);
  ClientId startClient(int32_t listeningPort);
private:
  zmq::context_t &context_;
  zmq::socket_t socket_;

  void sendClientStartRequest(int32_t port);
};

// Bind to a socket, expecting a connection

#endif // CLIENTMANAGERINTERFACE_HPP_