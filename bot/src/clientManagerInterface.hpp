#ifndef CLIENTMANAGERINTERFACE_HPP_
#define CLIENTMANAGERINTERFACE_HPP_

#include <zmq.hpp>

#include <atomic>
#include <chrono>
#include <deque>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <mutex>
#include <thread>

class ClientManagerInterface {
public:
  using ClientId = int32_t;

  ClientManagerInterface(zmq::context_t &context);
  ~ClientManagerInterface();
  void runAsync();
  ClientId startClient(int32_t listeningPort);
  // void killClient(ClientId);
private:
  struct ClientOpenRequest {
    int32_t port;
    std::promise<ClientId> completedPromise;
  };

  static constexpr std::chrono::milliseconds kMaxHeartbeatSilence{100};
  zmq::context_t &context_;
  zmq::socket_t socket_;
  std::atomic<bool> running_{false};
  std::atomic<bool> shouldStop_{false};
  std::thread runThread_;
  std::mutex mutex_;
  std::condition_variable conditionVariable_;
  std::deque<ClientOpenRequest> clientsPendingOpen_;
  std::chrono::time_point<std::chrono::high_resolution_clock> lastMessageSent_;

  void run();
  ClientId privateStartClient(int32_t listeningPort);
  void sendClientStartRequest(int32_t port);
  void sendHeartbeat();
  // void saveClientId(ClientId clientId);

  // static std::vector<ClientId> runningClients_;
  // static std::mutex runningClientListMutex_;
  // static void signalHandler(int signal);
};

// Bind to a socket, expecting a connection

#endif // CLIENTMANAGERINTERFACE_HPP_