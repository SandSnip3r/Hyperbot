#ifndef CLIENTMANAGERINTERFACE_HPP_
#define CLIENTMANAGERINTERFACE_HPP_

#include <zmq.hpp>

#include <cstdint>
// #include <mutex>
// #include <vector>

class ClientManagerInterface {
public:
  using ClientId = int32_t;

  ClientManagerInterface(zmq::context_t &context);
  ClientId startClient(int32_t listeningPort);
  // void killClient(ClientId);
private:
  zmq::context_t &context_;
  zmq::socket_t socket_;

  void sendClientStartRequest(int32_t port);
  // void saveClientId(ClientId clientId);

  // static std::vector<ClientId> runningClients_;
  // static std::mutex runningClientListMutex_;
  // static void signalHandler(int signal);
};

// Bind to a socket, expecting a connection

#endif // CLIENTMANAGERINTERFACE_HPP_