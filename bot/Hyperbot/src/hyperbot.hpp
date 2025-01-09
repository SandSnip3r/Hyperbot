#ifndef HYPERBOT_HPP_
#define HYPERBOT_HPP_

#include "broker/eventBroker.hpp"
#include "config/serverConfig.hpp"
#include "pk2/gameData.hpp"

#include <string>

class Hyperbot {
public:
  void run();
private:
  config::ServerConfig serverConfig_;
  broker::EventBroker eventBroker_;
  pk2::GameData gameData_;

  void parseConfig();
  void initializeGameData();
};

#endif // HYPERBOT_HPP_