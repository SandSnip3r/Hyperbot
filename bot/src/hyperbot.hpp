#ifndef HYPERBOT_HPP_
#define HYPERBOT_HPP_

#include "broker/eventBroker.hpp"
#include "config/serverConfig.hpp"

#include <silkroad_lib/pk2/gameData.hpp>

class Hyperbot {
public:
  ~Hyperbot();
  void run();
private:
  config::ServerConfig serverConfig_;
  broker::EventBroker eventBroker_;
  sro::pk2::GameData gameData_;

  void parseConfig();
  void initializeGameData();
};

#endif // HYPERBOT_HPP_
