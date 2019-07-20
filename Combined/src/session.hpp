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
	const std::string kSilkroadPath_;
  const uint8_t kLocale_;
  Loader loader_;
  BrokerSystem broker_;
  Proxy proxy_{broker_, Config::BindPort};
  Bot bot_{broker_};
public:
  Session(const std::string &silkroadPath, uint8_t locale);
  ~Session();
  void start();
};

#endif