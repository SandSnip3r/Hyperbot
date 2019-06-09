#include "bot.hpp"
#include "brokerSystem.hpp"
#include "loader.hpp"
#include "proxy.hpp"

#include <functional>
#include <string>

#ifndef SESSION_HPP_
#define SESSION_HPP_

class Session {
  // This is a facilitator that starts the proxy, starts the client, redirects the client, and contains intelligence in the bot
private:
	const std::string kSilkroadPath = "C:\\Program Files (x86)\\Evolin\\";
  const std::string kConfigPath_;
  Loader loader_{kSilkroadPath};
  BrokerSystem broker_;
  Proxy proxy_{broker_, Config::BindPort};
  Bot bot_{broker_};
public:
  Session(const std::string &configPath);
  ~Session();
  void start();
};

#endif