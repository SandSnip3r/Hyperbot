#include "login.hpp"

#include "bot.hpp"
#include "event/event.hpp"
#include "packet/building/clientAgentAuthRequest.hpp"
#include "packet/building/clientAgentCharacterSelectionJoinRequest.hpp"
#include "packet/building/clientGatewayLoginIbuvAnswer.hpp"
#include "packet/building/clientGatewayLoginRequest.hpp"
#include "packet/building/clientGatewayShardListRequest.hpp"

#include <absl/log/log.h>
#include <absl/strings/str_format.h>
#include <absl/strings/str_join.h>

namespace state::machine {

Login::Login(Bot &bot, const CharacterLoginInfo &characterLoginInfo) : StateMachine(bot), username_(characterLoginInfo.username), password_(characterLoginInfo.password), characterName_(characterLoginInfo.characterName) {
  VLOG(1) << absl::StreamFormat("Constructed Login state machine for character %s", characterName_);
}

Login::Login(StateMachine *parent, const CharacterLoginInfo &characterLoginInfo) : StateMachine(parent), username_(characterLoginInfo.username), password_(characterLoginInfo.password), characterName_(characterLoginInfo.characterName) {
  VLOG(1) << absl::StreamFormat("Constructed Login state machine for character %s", characterName_);
}

Login::~Login() {}

Status Login::onUpdate(const event::Event *event) {
  if (!initialized_) {
    initialized_ = true;
    // The client will try to send the agent auth request (after we already sent the gateway auth request).
    // TODO: Once we have clientless, blocking this packet is not necessary (but also not harmful).
    pushBlockedOpcode(packet::Opcode::kClientAgentAuthRequest);

    if (bot_.worldState().shardListResponse_) {
      VLOG(1) << absl::StreamFormat("[%s] Already have shard list", characterName_);
      waitingOnShardList_ = false;
    } else {
      VLOG(1) << absl::StreamFormat("[%s] Don't yet have a shard list, we'll get it ourself", characterName_);
      // Since this state machine is doing the login, we send these two packets. Normally the client sends them. If instead the user wants to manually log in using the client, we should let the client send them.
      pushBlockedOpcode(packet::Opcode::kClientGatewayShardListRequest);
      pushBlockedOpcode(packet::Opcode::kClientGatewayShardListPingRequest);
    }
  }

  if (!waitingOnShardList_ && !bot_.worldState().shardListResponse_.has_value()) {
    // Don't yet have a shard list, request it.
    VLOG(1) << absl::StreamFormat("[%s] Requesting shard list", characterName_);
    injectPacket(packet::building::ClientGatewayShardListRequest::packet(), PacketContainer::Direction::kBotToServer);
    waitingOnShardList_ = true;
    return Status::kNotDone;
  }

  // Everything else in this function expects an event.
  if (event == nullptr) {
    // No event, nothing to do.
    VLOG(3) << absl::StreamFormat("[%s] No event, nothing to do.", characterName_);
    return Status::kNotDone;
  }

  // Anything to do with logging in will be a session specific event.
  if (const auto *sessionSpecificEvent = dynamic_cast<const event::SessionSpecificEvent*>(event); sessionSpecificEvent != nullptr) {
    if (sessionSpecificEvent->sessionId != bot_.sessionId()) {
      // Not for us.
      VLOG(3) << absl::StreamFormat("[%s] Session specific event for someone else", characterName_);
      return Status::kNotDone;
    }

    if (const auto *shardListReceivedEvent = dynamic_cast<const event::ShardListReceived*>(sessionSpecificEvent); shardListReceivedEvent != nullptr) {
      VLOG(1) << absl::StreamFormat("[%s] Received shard list for our session (session ID: %d)", characterName_, shardListReceivedEvent->sessionId);
      if (!loginRequestSent_) {
        // This shard list is for our session. Try to log in.
        // TODO: For now, we just choose the first shard.
        if (shardListReceivedEvent->shards.empty()) {
          throw std::runtime_error("Received shard list, but it is empty");
        }
        if (shardListReceivedEvent->shards.size() > 1) {
          LOG(WARNING) << absl::StreamFormat("[%s] There are multiple shards, picking the first one", characterName_);
        }
        VLOG(1) << absl::StreamFormat("[%s] Trying to login with username \"%s\" and password \"%s\", to shard ID %d", characterName_, username_, password_, shardListReceivedEvent->shards.at(0).shardId);
        const auto loginAuthPacket = packet::building::ClientGatewayLoginRequest::packet(bot_.gameData().divisionInfo().locale, username_, password_, shardListReceivedEvent->shards.at(0).shardId);
        injectPacket(loginAuthPacket, PacketContainer::Direction::kBotToServer);
        waitingOnShardList_ = false;
        loginRequestSent_ = true;
      } else {
        VLOG(1) << absl::StreamFormat("[%s] Already sent login request, not resending", characterName_);
      }
    }

    if (waitingOnShardList_) {
      // Have not yet received the shard list.
      VLOG(1) << absl::StreamFormat("[%s] Still waiting on shard list", characterName_);
      return Status::kNotDone;
    }

    if (const auto *gatewayLoginResponseReceived = dynamic_cast<const event::GatewayLoginResponseReceived*>(sessionSpecificEvent); gatewayLoginResponseReceived != nullptr) {
      VLOG(1) << absl::StreamFormat("[%s] Got agentserver login token: %d", characterName_, gatewayLoginResponseReceived->agentServerToken);
      agentServerToken_ = gatewayLoginResponseReceived->agentServerToken;
    } else if (const auto *connectedToAgentServerEvent = dynamic_cast<const event::ConnectedToAgentServer*>(sessionSpecificEvent); connectedToAgentServerEvent != nullptr) {
      VLOG(1) << absl::StreamFormat("[%s] Connected to agentserver", characterName_);
      // Send our auth packet.
      if (!agentServerToken_) {
        throw std::runtime_error("Connected to agentserver but don't have token");
      }
      const auto clientAuthPacket = packet::building::ClientAgentAuthRequest::packet(*agentServerToken_, username_, password_, bot_.gameData().divisionInfo().locale, kMacAddress);
      injectPacket(clientAuthPacket, PacketContainer::Direction::kBotToServer);
    } else if (const auto *characterListReceivedEvent = dynamic_cast<const event::CharacterListReceived*>(sessionSpecificEvent); characterListReceivedEvent != nullptr) {
      VLOG(1) << absl::StreamFormat("[%s] Received character list for our session: [ %s ]", characterName_, absl::StrJoin(characterListReceivedEvent->characters, ", ", [](std::string *out, const auto character){
        out->append(character.name);
      }));

      // Search for our character in the character list
      auto it = std::find_if(characterListReceivedEvent->characters.begin(), characterListReceivedEvent->characters.end(), [this](const packet::structures::character_selection::Character &character) {
        return character.name == characterName_;
      });
      if (it == characterListReceivedEvent->characters.end()) {
        LOG(WARNING) << absl::StreamFormat("[%s] Unable to find character", characterName_);
        return Status::kDone;
      }

      // Found our character, select it
      VLOG(1) << absl::StreamFormat("[%s] Selecting character", characterName_);
      const auto charSelectionPacket = packet::building::ClientAgentCharacterSelectionJoinRequest::packet(characterName_);
      injectPacket(charSelectionPacket, PacketContainer::Direction::kBotToServer);
    } else if (const auto *ibuvChallengeReceivedEvent = dynamic_cast<const event::IbuvChallengeReceived*>(sessionSpecificEvent); ibuvChallengeReceivedEvent != nullptr) {
      VLOG(1) << absl::StreamFormat("[%s] Got captcha. Sending answer", characterName_);
      const auto captchaAnswerPacket = packet::building::ClientGatewayLoginIbuvAnswer::packet(kCaptchaAnswer);
      injectPacket(captchaAnswerPacket, PacketContainer::Direction::kBotToServer);
    } else if (const auto *serverAuthSuccessEvent = dynamic_cast<const event::ServerAuthSuccess*>(sessionSpecificEvent); serverAuthSuccessEvent != nullptr) {
      VLOG(1) << absl::StreamFormat("[%s] Successfully logged in.", characterName_);
    } else if (const auto *characterSelectionJoinSuccessEvent = dynamic_cast<const event::CharacterSelectionJoinSuccess*>(sessionSpecificEvent); characterSelectionJoinSuccessEvent != nullptr) {
      VLOG(1) << absl::StreamFormat("[%s] Successfully selected character", characterName_);
    } else if (const auto *selfSpawnedEvent = dynamic_cast<const event::SelfSpawned*>(sessionSpecificEvent); selfSpawnedEvent != nullptr) {
      VLOG(1) << absl::StreamFormat("[%s] Self spawned", characterName_);
      spawned_ = true;
    }
  } else if (const auto *bodyStateChangedEvent = dynamic_cast<const event::EntityBodyStateChanged*>(event); bodyStateChangedEvent != nullptr) {
    if (spawned_ && bodyStateChangedEvent->globalId == bot_.selfState()->globalId) {
      // Our body state changed. This is the last event as part of the character-spawning process.
      VLOG(1) << absl::StreamFormat("[%s] Body state changed. Login state machine complete.", characterName_);
      return Status::kDone;
    }
  }
  return Status::kNotDone;
}

} // namespace state::machine