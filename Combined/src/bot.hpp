#ifndef BOT_HPP_
#define BOT_HPP_

#include "packetProcessor.hpp"
#include "broker/eventBroker.hpp"
#include "broker/packetBroker.hpp"
#include "config/characterLoginData.hpp"
#include "module/characterInfoModule.hpp"
#include "packet/parsing/packetParser.hpp"
#include "pk2/gameData.hpp"
#include "state/entity.hpp"
#include "state/self.hpp"
#include "ui/userInterface.hpp"

class Bot {
public:
  Bot(const config::CharacterLoginData &loginData,
      const pk2::GameData &gameData,
      broker::PacketBroker &broker);

protected:
  friend class broker::EventBroker;
  void handleEvent(const event::Event *event);

private:
  const config::CharacterLoginData &loginData_; // TODO: Move this into a configuration object
  const pk2::GameData &gameData_;
  state::Entity entityState_;
  state::Self selfState_{gameData_};
  broker::PacketBroker &broker_;
  broker::EventBroker eventBroker_;
  ui::UserInterface userInterface_{eventBroker_};
  packet::parsing::PacketParser packetParser_{gameData_};
  PacketProcessor packetProcessor_{entityState_, selfState_, broker_, eventBroker_, userInterface_, packetParser_, gameData_};
  module::CharacterInfoModule characterInfoModule_{entityState_, selfState_, broker_, eventBroker_, userInterface_, packetParser_, gameData_};

  void subscribeToEvents();
  // Login events
  void handleStateShardIdUpdated() const;
  void handleStateConnectedToAgentServerUpdated();
  void handleStateReceivedCaptchaPromptUpdated() const;
  void handleStateCharacterListUpdated() const;
  // Movement events
  void handleMovementEnded();
  void handleSpeedUpdated();
};

#endif