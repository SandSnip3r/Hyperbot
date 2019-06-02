#include "bot.hpp"
#include "loader.hpp"
#include "proxy.hpp"

#include <functional>
#include <string>

#ifndef SESSION_HPP_
#define SESSION_HPP_

class Session {
  //This is a facilitator that starts the proxy, starts the client, redirects the client, and runs the bot
private:
	const std::string kSilkroadPath = "C:\\Program Files (x86)\\Evolin\\";
  const std::string kConfigPath_;
  Loader loader_{kSilkroadPath};

  //Construct the Bot with a function pointer so that it has access to inject packets using the Proxy
  Bot bot_{std::bind(&Proxy::inject, &proxy_, std::placeholders::_1, std::placeholders::_2)};

  //Construct the Proxy with a function pointer so that it has access to send incoming packets to the Bot
  Proxy proxy_{std::bind(&Bot::packetReceived, &bot_, std::placeholders::_1, std::placeholders::_2), Config::BindPort};
public:
  Session(const std::string &configPath);
  ~Session();
  void start();
};

#endif