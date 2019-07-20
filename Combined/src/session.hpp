#ifndef SESSION_HPP_
#define SESSION_HPP_

#include "bot.hpp"
#include "brokerSystem.hpp"
#include "gameData.hpp"
#include "loader.hpp"
#include "proxy.hpp"

#include <functional>
#include <string>

namespace {
namespace fs = std::experimental::filesystem::v1;
}

class Session {
  // This is a facilitator that starts the proxy, starts the client, redirects the client, and contains intelligence in the bot
public:
  Session(const pk2::media::GameData &gameData, const fs::path &kSilkroadDirectoryPath);
  ~Session();
  void start();
private:
  const pk2::media::GameData &gameData_;
  Loader loader_;
  BrokerSystem broker_;
  Proxy proxy_{broker_, Config::BindPort};
  Bot bot_{broker_};
};

#endif