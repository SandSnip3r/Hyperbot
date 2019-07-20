#ifndef SESSION_HPP_
#define SESSION_HPP_

#include "bot.hpp"
#include "brokerSystem.hpp"
#include "configData.hpp"
#include "gameData.hpp"
#include "loader.hpp"
#include "proxy.hpp"

#include <functional>
#include <string>

class Session {
  // This is a facilitator that starts the proxy, starts the client, redirects the client, and contains intelligence in the bot
public:
  Session(const pk2::media::GameData &gameData,
          const std::experimental::filesystem::v1::path &kSilkroadDirectoryPath,
          const config::CharacterLoginData &loginData);
  ~Session();
  void start();
private:
  const pk2::media::GameData &gameData_;
  const std::experimental::filesystem::v1::path &kSilkroadDirectoryPath_;
  const config::CharacterLoginData &loginData_;
  Loader loader_{kSilkroadDirectoryPath_, gameData_.divisionInfo()};
  BrokerSystem broker_;
  Proxy proxy_{gameData_, broker_, Config::BindPort};
  Bot bot_{loginData_, gameData_.itemData(), gameData_.skillData(), broker_};
};

#endif