#ifndef SESSION_HPP_
#define SESSION_HPP_

#include "bot.hpp"
#include "characterLoginInfo.hpp"
#include "clientManagerInterface.hpp"
#include "common/sessionId.hpp"
#include "pk2/gameData.hpp"
#include "proxy.hpp"
#include "packetProcessor.hpp"
#include "state/worldState.hpp"

namespace ui {
class RlUserInterface;
} // namespace ui

#include <atomic>
#include <functional>
#include <optional>
#include <string>

// Session is a facilitator that:
//  - Starts the Proxy
//  - Starts the client and redirects its connection to Proxy
//  - Acts upon packets by the intelligence in Bot
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
          ClientManagerInterface &clientManagerInterface,
          ui::RlUserInterface &rlUserInterface);
  ~Session() = default;
  void setCharacter(const CharacterLoginInfo &characterLoginInfo);
  void initialize();
  void runAsync();
  const state::WorldState& getWorldState() const;
  Bot& getBot();
  SessionId sessionId() const { return sessionId_; }

  std::future<void> asyncOpenClient();
  std::future<void> connectClientlessAsync();
private:
  SessionId sessionId_{createUniqueSessionId()};
  bool initialized_{false};
  const pk2::GameData &gameData_;
  broker::EventBroker &eventBroker_;
  state::WorldState &worldState_;
  ui::RlUserInterface &rlUserInterface_;
  PacketProcessor packetProcessor_{sessionId_, worldState_, eventBroker_, gameData_};
  Proxy proxy_{gameData_, packetProcessor_};
  Bot bot_{sessionId_, gameData_, proxy_, eventBroker_, worldState_, rlUserInterface_};
  ClientManagerInterface &clientManagerInterface_;
  std::optional<ClientManagerInterface::ClientId> clientId_;

  void handleClientDiedEvent(const event::Event *event);

  static std::atomic<SessionId> nextSessionId;
  static SessionId createUniqueSessionId();
};

#endif
