#ifndef SESSION_HPP_
#define SESSION_HPP_

#include "bot.hpp"
#include "proxy.hpp"
#include "sessionId.hpp"

#if defined(_WIN32)
#include "loader.hpp"
#endif

#include "broker/packetBroker.hpp"
#include "pk2/gameData.hpp"
#include "state/worldState.hpp"

#include <atomic>
#include <functional>
#include <string>
#include <string_view>

// Session is a facilitator that:
//  - Starts the Proxy
//  - Starts the client and redirects its connection to Proxy using Loader
//  - Acts upon packets by the intelligence in Bot
// PacketBroker is the communication channel between Bot and Proxy
class Session {
public:
  Session(const pk2::GameData &gameData,
          std::string_view clientPath,
          broker::EventBroker &eventBroker,
          state::WorldState &worldState);
  ~Session() = default;
  void setCharacterToLogin(std::string_view characterName);
  void initialize();
  void runAsync();
  const state::WorldState& getWorldState() const;
private:
  SessionId sessionId_{createUniqueSessionId()};
  bool initialized_{false};
  const pk2::GameData &gameData_;
  broker::EventBroker &eventBroker_;
#if defined(_WIN32)
  Loader loader_;
#endif
  broker::PacketBroker packetBroker_;
  Proxy proxy_{gameData_, packetBroker_};
  Bot bot_;

  static std::atomic<SessionId> nextSessionId;
  static SessionId createUniqueSessionId();
};

#endif