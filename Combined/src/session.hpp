#ifndef SESSION_HPP_
#define SESSION_HPP_

#include "bot.hpp"
#include "loader.hpp"
#include "proxy.hpp"

#include "broker/packetBroker.hpp"
#include "config/config.hpp"
#include "pk2/gameData.hpp"

#include <functional>
#include <string>

// Session is a facilitator that:
//  - Starts the Proxy
//  - Starts the client and redirects its connection to Proxy using Loader
//  - Acts upon packets by the intelligence in Bot
// PacketBroker is the communication channel between Bot and Proxy
class Session {
public:
  Session(const pk2::GameData &gameData,
          const config::Config &config,
          broker::EventBroker &eventBroker);
  ~Session();
  void initialize();
  void run();
  const state::WorldState& getWorldState() const;
private:
  const pk2::GameData &gameData_;
  const config::Config &config_;
  broker::EventBroker &eventBroker_;
  Loader loader_{config_, gameData_.divisionInfo()};
  broker::PacketBroker packetBroker_;
  Proxy proxy_{gameData_, packetBroker_};
  Bot bot_{config_, gameData_, proxy_, packetBroker_, eventBroker_};
};

#endif