#ifndef SESSION_HPP_
#define SESSION_HPP_

#include "bot.hpp"
#include "characterLoginInfo.hpp"
#include "clientManagerInterface.hpp"
#include "proxy.hpp"
#include "sessionId.hpp"

#include "broker/packetBroker.hpp"
#include "pk2/gameData.hpp"
#include "state/worldState.hpp"

#include <atomic>
#include <functional>
#include <optional>
#include <string>

// Session is a facilitator that:
//  - Starts the Proxy
//  - Starts the client and redirects its connection to Proxy
//  - Acts upon packets by the intelligence in Bot
// PacketBroker is the communication channel between Bot and Proxy
//
// Proper invocation order (repeated numbers mean order does not matter):
//  1. Session()
//  2. initialize()
//    - Primarily subscribes to events/packets
//  2. setCharacter()
//  3. runAsync()
class Session {
public:
  Session(const pk2::GameData &gameData,
          broker::EventBroker &eventBroker,
          state::WorldState &worldState,
          ClientManagerInterface &clientManagerInterface);
  ~Session() = default;
  void setCharacter(const CharacterLoginInfo &characterLoginInfo);
  void initialize();
  void runAsync();
  const state::WorldState& getWorldState() const;
  Bot& getBot();
private:
  SessionId sessionId_{createUniqueSessionId()};
  bool initialized_{false};
  const pk2::GameData &gameData_;
  broker::EventBroker &eventBroker_;
  broker::PacketBroker packetBroker_;
  Proxy proxy_{gameData_, packetBroker_};
  Bot bot_;
  ClientManagerInterface &clientManagerInterface_;
  std::optional<ClientManagerInterface::ClientId> clientId_;

  static std::atomic<SessionId> nextSessionId;
  static SessionId createUniqueSessionId();
};

#endif