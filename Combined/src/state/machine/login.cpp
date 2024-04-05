#include "login.hpp"

#include "bot.hpp"
#include "event/event.hpp"
#include "logging.hpp"
// #include "packet/building/clientAgentInventoryItemUseRequest.hpp"
#include "packet/building/clientAgentAuthRequest.hpp"
#include "packet/building/clientAgentCharacterSelectionJoinRequest.hpp"
#include "packet/building/clientGatewayLoginIbuvAnswer.hpp"
#include "packet/building/clientGatewayLoginRequest.hpp"

// #include <stdexcept>

namespace state::machine {

Login::Login(Bot &bot, const std::string &username, const std::string &password, const std::string &characterName) : StateMachine(bot), username_(username), password_(password), characterName_(characterName) {
  stateMachineCreated(kName);
  // The client will try to send the agent auth request (after we already sent the gateway auth request).
  // TODO: Once we have clientless, blocking this packet is not neccessary (but also not harmful).
  pushBlockedOpcode(packet::Opcode::kClientAgentAuthRequest);
}

Login::~Login() {
  stateMachineDestroyed();
}

void Login::onUpdate(const event::Event *event) {
  if (event) {
    if (event->eventCode == event::EventCode::kStateShardIdUpdated) {
      // We received the server list from the server. Try to log in.
      const auto loginAuthPacket = packet::building::ClientGatewayLoginRequest::packet(bot_.gameData().divisionInfo().locale, username_, password_, bot_.selfState().shardId);
      bot_.packetBroker().injectPacket(loginAuthPacket, PacketContainer::Direction::kClientToServer);
    } else if (event->eventCode == event::EventCode::kStateConnectedToAgentServerUpdated) {
      // Send our auth packet.
      const auto clientAuthPacket = packet::building::ClientAgentAuthRequest::packet(bot_.selfState().token, username_, password_, bot_.gameData().divisionInfo().locale, kMacAddress);
      bot_.packetBroker().injectPacket(clientAuthPacket, PacketContainer::Direction::kClientToServer);
    } else if (event->eventCode == event::EventCode::kStateCharacterListUpdated) {
      LOG() << "Char list received: [ ";
      for (const auto &i : bot_.selfState().characterList) {
        std::cout << i.name << ' ';
      }
      std::cout << "]\n";

      // Search for our character in the character list
      auto it = std::find_if(bot_.selfState().characterList.begin(), bot_.selfState().characterList.end(), [this](const packet::structures::CharacterSelection::Character &character) {
        return character.name == characterName_;
      });
      if (it == bot_.selfState().characterList.end()) {
        LOG() << "Unable to find character \"" << characterName_ << "\"" << std::endl;
        return;
      }

      // Found our character, select it
      LOG() << "Selecting \"" << characterName_ << "\"" << std::endl;
      auto charSelectionPacket = packet::building::ClientAgentCharacterSelectionJoinRequest::packet(characterName_);
      bot_.packetBroker().injectPacket(charSelectionPacket, PacketContainer::Direction::kClientToServer);
    } else if (event->eventCode == event::EventCode::kStateReceivedCaptchaPromptUpdated) {
      LOG() << "Got captcha. Sending answer\n";
      const auto captchaAnswerPacket = packet::building::ClientGatewayLoginIbuvAnswer::packet(kCaptchaAnswer);
      bot_.packetBroker().injectPacket(captchaAnswerPacket, PacketContainer::Direction::kClientToServer);
    } else if (event->eventCode == event::EventCode::kLoggedIn) {
      LOG() << "Logged in." << std::endl;
    } else if (event->eventCode == event::EventCode::kSpawned) {
      LOG() << "Logged in and character selected. Done." << std::endl;
      done_ = true;
    }
  }
}

bool Login::done() const {
  return done_;
}

} // namespace state::machine