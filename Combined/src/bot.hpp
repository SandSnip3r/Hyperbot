#ifndef BOT_HPP_
#define BOT_HPP_

#include "broker/eventBroker.hpp"
#include "broker/packetBroker.hpp"
#include "config/characterLoginData.hpp"
#include "module/characterInfoModule.hpp"
#include "module/loginModule.hpp"
#include "module/movementModule.hpp"
#include "module/skillUseModule.hpp"
#include "packet/parsing/packetParser.hpp"
#include "pk2/gameData.hpp"
#include "state/entity.hpp"
#include "state/self.hpp"

class Bot {
public:
  Bot(const config::CharacterLoginData &loginData,
      const pk2::GameData &gameData,
      pathfinder::Pathfinder &pathfinder,
      broker::PacketBroker &broker);
private:
  const config::CharacterLoginData &loginData_;
  const pk2::GameData &gameData_;
  pathfinder::Pathfinder &pathfinder_;
  state::Entity entityState_;
  state::Self selfState_{gameData_};
  storage::Storage inventory_;
  broker::PacketBroker &broker_;
  broker::EventBroker eventBroker_;
  packet::parsing::PacketParser packetParser_{gameData_};
  module::CharacterInfoModule characterInfoModule_{entityState_, selfState_, inventory_, broker_, eventBroker_, packetParser_, gameData_};
  module::LoginModule loginModule_{broker_, packetParser_, loginData_, gameData_.divisionInfo()};
  module::MovementModule movementModule_{entityState_, selfState_, inventory_, broker_, eventBroker_, packetParser_, gameData_, pathfinder_};
  module::SkillUseModule skillUseModule_{entityState_, selfState_, inventory_, broker_, eventBroker_, packetParser_, gameData_};
};

#endif