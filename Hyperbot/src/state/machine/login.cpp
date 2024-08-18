#include "login.hpp"

#include "bot.hpp"
#include "event/event.hpp"
#include "packet/building/clientAgentAuthRequest.hpp"
#include "packet/building/clientAgentCharacterSelectionJoinRequest.hpp"
#include "packet/building/clientGatewayLoginIbuvAnswer.hpp"
#include "packet/building/clientGatewayLoginRequest.hpp"

#include <absl/log/log.h>
#include <absl/strings/str_format.h>
#include <absl/strings/str_join.h>

namespace state::machine {

Login::Login(Bot &bot, std::string_view username, std::string_view password, std::string_view characterName) : StateMachine(bot), username_(username), password_(password), characterName_(characterName) {
  stateMachineCreated(kName);
  VLOG(1) << absl::StreamFormat("Constructed Login state machine for character %s", characterName_);
  // The client will try to send the agent auth request (after we already sent the gateway auth request).
  // TODO: Once we have clientless, blocking this packet is not necessary (but also not harmful).
  pushBlockedOpcode(packet::Opcode::kClientAgentAuthRequest);
}

Login::~Login() {
  stateMachineDestroyed();
}

void Login::onUpdate(const event::Event *event) {
  if (event) {
    if (const auto *shardListReceivedEvent = dynamic_cast<const event::ShardListReceived*>(event); shardListReceivedEvent != nullptr) {
      VLOG(1) << absl::StreamFormat("[%s]", characterName_);
      if (shardListReceivedEvent->sessionId == bot_.sessionId()) {
        // This shard list is for our session. Try to log in.
        // TODO: For now, we just choose the first shard.
        if (shardListReceivedEvent->shards.empty()) {
          throw std::runtime_error("Received shard list, but it is empty");
        }
        if (shardListReceivedEvent->shards.size() > 1) {
          LOG(WARNING) << "There are multiple shards, picking the first one";
        }
        VLOG(2) << "Received shard list for our session (session ID: " << shardListReceivedEvent->sessionId << ")";
        VLOG(1) << absl::StreamFormat("Trying to login with username \"%s\" and password \"%s\", to shard ID %d", username_, password_, shardListReceivedEvent->shards.at(0).shardId);
        const auto loginAuthPacket = packet::building::ClientGatewayLoginRequest::packet(bot_.gameData().divisionInfo().locale, username_, password_, shardListReceivedEvent->shards.at(0).shardId);
        bot_.packetBroker().injectPacket(loginAuthPacket, PacketContainer::Direction::kBotToServer);
      }
    } else if (const auto *gatewayLoginResponseReceived = dynamic_cast<const event::GatewayLoginResponseReceived*>(event); gatewayLoginResponseReceived != nullptr) {
      VLOG(1) << absl::StreamFormat("[%s]", characterName_);
      if (gatewayLoginResponseReceived->sessionId == bot_.sessionId()) {
        VLOG(2) << "Got agentserver login token for our session: " << gatewayLoginResponseReceived->agentServerToken;
        agentServerToken_ = gatewayLoginResponseReceived->agentServerToken;
      }
    } else if (const auto *connectedToAgentServerEvent = dynamic_cast<const event::ConnectedToAgentServer*>(event); connectedToAgentServerEvent != nullptr) {
      VLOG(1) << absl::StreamFormat("[%s]", characterName_);
      if (connectedToAgentServerEvent->sessionId == bot_.sessionId()) {
        VLOG(2) << "Connected to agentserver";
        // Send our auth packet.
        if (!agentServerToken_) {
          throw std::runtime_error("Connected to agentserver but don't have token");
        }
        const auto clientAuthPacket = packet::building::ClientAgentAuthRequest::packet(*agentServerToken_, username_, password_, bot_.gameData().divisionInfo().locale, kMacAddress);
        bot_.packetBroker().injectPacket(clientAuthPacket, PacketContainer::Direction::kBotToServer);
      }
    } else if (const auto *characterListReceivedEvent = dynamic_cast<const event::CharacterListReceived*>(event); characterListReceivedEvent != nullptr) {
      VLOG(1) << absl::StreamFormat("[%s]", characterName_);
      if (characterListReceivedEvent->sessionId == bot_.sessionId()) {
        VLOG(1) << absl::StreamFormat("Received character list for our session: [ %s ]", absl::StrJoin(characterListReceivedEvent->characters, ", ", [](std::string *out, const auto character){
          out->append(character.name);
        }));

        // Search for our character in the character list
        auto it = std::find_if(characterListReceivedEvent->characters.begin(), characterListReceivedEvent->characters.end(), [this](const packet::structures::character_selection::Character &character) {
          return character.name == characterName_;
        });
        if (it == characterListReceivedEvent->characters.end()) {
          LOG(WARNING) << "Unable to find character \"" << characterName_ << "\"";
          done_ = true;
          return;
        }

        // Found our character, select it
        VLOG(1) << "Selecting \"" << characterName_ << "\"";
        const auto charSelectionPacket = packet::building::ClientAgentCharacterSelectionJoinRequest::packet(characterName_);
        bot_.packetBroker().injectPacket(charSelectionPacket, PacketContainer::Direction::kBotToServer);
      }
    } else if (const auto *ibuvChallengeReceivedEvent = dynamic_cast<const event::IbuvChallengeReceived*>(event); ibuvChallengeReceivedEvent != nullptr) {
      VLOG(1) << absl::StreamFormat("[%s]", characterName_);
      if (ibuvChallengeReceivedEvent->sessionId == bot_.sessionId()) {
        VLOG(2) << "Got captcha. Sending answer";
        const auto captchaAnswerPacket = packet::building::ClientGatewayLoginIbuvAnswer::packet(kCaptchaAnswer);
        bot_.packetBroker().injectPacket(captchaAnswerPacket, PacketContainer::Direction::kBotToServer);
      }
    } else if (const auto *serverAuthSuccessEvent = dynamic_cast<const event::ServerAuthSuccess*>(event); serverAuthSuccessEvent != nullptr) {
      VLOG(1) << absl::StreamFormat("[%s]", characterName_);
      if (serverAuthSuccessEvent->sessionId == bot_.sessionId()) {
        VLOG(2) << "Successfully logged in.";
      }
    } else if (const auto *characterSelectionJoinSuccessEvent = dynamic_cast<const event::CharacterSelectionJoinSuccess*>(event); characterSelectionJoinSuccessEvent != nullptr) {
      VLOG(1) << absl::StreamFormat("[%s]", characterName_);
      if (characterSelectionJoinSuccessEvent->sessionId == bot_.sessionId()) {
        VLOG(1) << "Successfully selected character. Login complete.";
        done_ = true;
      }
    }
  }
}

bool Login::done() const {
  return done_;
}

} // namespace state::machine