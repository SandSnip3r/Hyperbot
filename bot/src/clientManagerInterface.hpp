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

namespace broker {
class EventBroker;
} // namespace broker

class ClientManagerInterface {
public:
  using ClientId = int32_t;

  ClientManagerInterface(zmq::context_t &context, broker::EventBroker &eventBroker);
  ~ClientManagerInterface();
  void runAsync();
  ClientId startClient(int32_t listeningPort);
private:
  struct ClientOpenRequest {
    int32_t port;
    std::promise<ClientId> completedPromise;
  };

  static constexpr std::chrono::milliseconds kMaxHeartbeatSilence{100};
  zmq::context_t &context_;
  broker::EventBroker &eventBroker_;
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
};

#endif // CLIENTMANAGERINTERFACE_HPP_