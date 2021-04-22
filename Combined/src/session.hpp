#ifndef SESSION_HPP_
#define SESSION_HPP_

#include "bot.hpp"
#include "loader.hpp"

#include "pathfinder.h"

#include "proxy.hpp"
#include "broker/packetBroker.hpp"
#include "config/configData.hpp"
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
          const std::filesystem::path &kSilkroadDirectoryPath,
          const config::CharacterLoginData &loginData);
  ~Session();
  void start();
private:
  const pk2::GameData &gameData_;
  const std::filesystem::path &kSilkroadDirectoryPath_;
  const config::CharacterLoginData &loginData_;
  const triangleio trash;
  Pathfinder pathfinder_{trash, trash};
  Loader loader_{kSilkroadDirectoryPath_, gameData_.divisionInfo()};
  broker::PacketBroker broker_;
  Proxy proxy_{gameData_, broker_};
  Bot bot_{loginData_, gameData_, broker_};
};

#endif