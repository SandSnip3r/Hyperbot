#include "bot.hpp"

#include "packet/building/clientAgentAuthRequest.hpp"
#include "packet/building/clientAgentCharacterSelectionJoinRequest.hpp"
#include "packet/building/clientGatewayLoginIbuvAnswer.hpp"
#include "packet/building/clientGatewayLoginRequest.hpp"

Bot::Bot(const config::CharacterLoginData &loginData,
         const pk2::GameData &gameData,
         broker::PacketBroker &broker) :
      loginData_(loginData),
      gameData_(gameData),
      broker_(broker) {
  eventBroker_.run();
  userInterface_.run();
  subscribeToEvents();
}

void Bot::subscribeToEvents() {
  auto eventHandleFunction = std::bind(&Bot::handleEvent, this, std::placeholders::_1);
  eventBroker_.subscribeToEvent(event::EventCode::kStateShardIdUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kStateConnectedToAgentServerUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kStateReceivedCaptchaPromptUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kStateCharacterListUpdated, eventHandleFunction);
}

void Bot::handleEvent(const event::Event *event) {
  std::unique_lock<std::mutex> selfStateLock(selfState_.selfMutex);

  const auto eventCode = event->eventCode;
  switch (eventCode) {
    case event::EventCode::kStateShardIdUpdated:
      handleStateShardIdUpdated();
      break;
    case event::EventCode::kStateConnectedToAgentServerUpdated:
      handleStateConnectedToAgentServerUpdated();
      break;
    case event::EventCode::kStateReceivedCaptchaPromptUpdated:
      handleStateReceivedCaptchaPromptUpdated();
      break;
    case event::EventCode::kStateCharacterListUpdated:
      handleStateCharacterListUpdated();
      break;
    default:
      std::cout << "Unhandled event subscribed to. Code:" << static_cast<int>(eventCode) << '\n';
      break;
  }
}

void Bot::handleStateShardIdUpdated() const {
  // We received the server list from the server, try to log in
  const auto loginAuthPacket = packet::building::ClientGatewayLoginRequest::packet(gameData_.divisionInfo().locale, loginData_.id, loginData_.password, selfState_.shardId);
  broker_.injectPacket(loginAuthPacket, PacketContainer::Direction::kClientToServer);
}

void Bot::handleStateConnectedToAgentServerUpdated() {
  const auto clientAuthPacket = packet::building::ClientAgentAuthRequest::packet(selfState_.token, loginData_.id, loginData_.password, gameData_.divisionInfo().locale, selfState_.kMacAddress);
  broker_.injectPacket(clientAuthPacket, PacketContainer::Direction::kClientToServer);
  // Set our state to logging in so that we'll know to block packets from the client if it tries to also login
  selfState_.loggingIn = true;
}

void Bot::handleStateReceivedCaptchaPromptUpdated() const {
  std::cout << "Got captcha. Sending answer\n";
  const auto captchaAnswerPacket = packet::building::ClientGatewayLoginIbuvAnswer::packet(selfState_.kCaptchaAnswer);
  broker_.injectPacket(captchaAnswerPacket, PacketContainer::Direction::kClientToServer);
}

void Bot::handleStateCharacterListUpdated() const {
  std::cout << "Char list received: [ ";
  for (const auto &i : selfState_.characterList) {
    std::cout << i.name << ' ';
  }
  std::cout << "]\n";

  // Search for our character in the character list
  auto it = std::find_if(selfState_.characterList.begin(), selfState_.characterList.end(), [this](const packet::structures::CharacterSelection::Character &character) {
    return character.name == loginData_.name;
  });
  if (it == selfState_.characterList.end()) {
    std::cout << "Unable to find character \"" << loginData_.name << "\"\n";
    return;
  }

  // Found our character, select it
  auto charSelectionPacket = packet::building::ClientAgentCharacterSelectionJoinRequest::packet(loginData_.name);
  broker_.injectPacket(charSelectionPacket, PacketContainer::Direction::kClientToServer);
}