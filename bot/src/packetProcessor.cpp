#include "packetProcessor.hpp"

#include "entity/entity.hpp"
#include "entity/item.hpp"
#include "entity/monster.hpp"
#include "helpers.hpp"
#include "packet/building/clientAgentActionCommandRequest.hpp"
#include "packet/opcode.hpp"
#include "type_id/categories.hpp"

#include <silkroad_lib/game_constants.hpp>
#include <silkroad_lib/position.hpp>
#include <silkroad_lib/position_math.hpp>

#include <tracy/Tracy.hpp>

#include <absl/algorithm/container.h>
#include <absl/container/flat_hash_set.h>
#include <absl/flags/flag.h>
#include <absl/log/log.h>
#include <absl/strings/str_format.h>
#include <absl/strings/str_join.h>

#include <fstream>

#define TRY_CAST_AND_HANDLE_PACKET(PACKET_TYPE, HANDLE_FUNCTION_NAME)        \
do {                                                                         \
  auto *castedParsedPacket = dynamic_cast<PACKET_TYPE*>(parsedPacket.get()); \
  if (castedParsedPacket != nullptr) {                                       \
    HANDLE_FUNCTION_NAME(*castedParsedPacket);                               \
    return;                                                                  \
  }                                                                          \
} while (false)

#define CHAR_LOG(severity) \
  LOG(severity) << characterNameForLog() << " "

#define CHAR_LOG_IF(severity, condition) \
  LOG_IF(severity, condition) << characterNameForLog() << " "

#define CHAR_VLOG(verbosity) \
  VLOG(verbosity) << characterNameForLog() << " "

ABSL_FLAG(bool, log_skills, false, "Log commands & skills");

WrappedCommand::WrappedCommand(const packet::structures::ActionCommand &command, const pk2::GameData &gameData) : actionCommand(command), gameData_(gameData) {

}

std::optional<std::string> WrappedCommand::skillName() const {
  if (actionCommand.commandType == packet::enums::CommandType::kExecute) {
    if (actionCommand.actionType == packet::enums::ActionType::kCast || actionCommand.actionType == packet::enums::ActionType::kDispel) {
      return gameData_.getSkillName(actionCommand.refSkillId);
    }
  }
  throw std::runtime_error("Asking for skill name for ActionCommand which does not use or cancel a skill");
}

std::ostream& operator<<(std::ostream &stream, const WrappedCommand &wrappedCommand) {
  stream << "CommandType: " << wrappedCommand.actionCommand.commandType;
  if (wrappedCommand.actionCommand.commandType == packet::enums::CommandType::kExecute) {
    stream << ", ActionType: " << wrappedCommand.actionCommand.actionType;
    if (wrappedCommand.actionCommand.actionType == packet::enums::ActionType::kCast || wrappedCommand.actionCommand.actionType == packet::enums::ActionType::kDispel) {
      stream << ", refSkillId: " << wrappedCommand.actionCommand.refSkillId;
      const auto maybeSkillName = wrappedCommand.skillName();
      if (maybeSkillName) {
        stream << ", skill name: \"" << *maybeSkillName << "\"";
      }
    }
    stream << ", targetType: " << wrappedCommand.actionCommand.targetType;
    if (wrappedCommand.actionCommand.targetType == packet::enums::TargetType::kEntity) {
      stream << ", targetGlobalId: " << wrappedCommand.actionCommand.targetGlobalId;
    } else if (wrappedCommand.actionCommand.targetType == packet::enums::TargetType::kLand) {
      stream << ", position: " << wrappedCommand.actionCommand.position;
    }
  }
  return stream;
}

namespace {

bool skillActionKilledTarget(sro::scalar_types::EntityGlobalId targetGlobalId, const packet::structures::SkillAction &skillAction) {
  if (targetGlobalId == 0) {
    // No target.
    return false;
  }
  for (const auto &hitObject : skillAction.hitObjects) {
    if (hitObject.targetGlobalId == targetGlobalId) {
      for (const auto &hit : hitObject.hits) {
        if (hit.hitResultFlag == packet::enums::HitResult::kKill) {
          // Killed our target.
          return true;
        }
      }
    }
  }
  return false;
}

} // anonymous namespace

PacketProcessor::PacketProcessor(SessionId sessionId,
                                 state::WorldState &worldState,
                                 broker::PacketBroker &brokerSystem,
                                 broker::EventBroker &eventBroker,
                                 const pk2::GameData &gameData) :
      sessionId_(sessionId),
      worldState_(worldState),
      packetBroker_(brokerSystem),
      eventBroker_(eventBroker),
      gameData_(gameData) {
}

void PacketProcessor::initialize() {
  subscribeToPackets();
}

std::shared_ptr<entity::Self> PacketProcessor::getSelfEntity() const {
  return selfEntity_;
}

void PacketProcessor::subscribeToPackets() {
  auto packetHandleFunction = std::bind(&PacketProcessor::handlePacket, this, std::placeholders::_1);

  // Server packets
  //   Login packets
  // packetBroker_.subscribeToClientPacket(packet::Opcode::kClientAgentAuthRequest, packetHandleFunction); // TODO: Do we want to see this packet?
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerGatewayPatchResponse, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerGatewayShardListResponse, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerGatewayLoginResponse, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kFrameworkMessageIdentify, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentAuthResponse, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentCharacterSelectionActionResponse, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentCharacterSelectionJoinResponse, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerGatewayLoginIbuvChallenge, packetHandleFunction);
  // packetBroker_.subscribeToServerPacket(static_cast<packet::Opcode>(0x6005), packetHandleFunction);
  //   Movement packets
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityUpdateAngle, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityUpdateMovement, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityUpdatePosition, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntitySyncPosition, packetHandleFunction);
  //   Character info packets
  packetBroker_.subscribeToClientPacket(packet::Opcode::kClientAgentInventoryOperationRequest, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentCharacterData, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentCosData, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentInventoryStorageData, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityUpdateHwanLevel, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityUpdateState, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityUpdateStatus, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityDamageEffect, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentAbnormalInfo, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentCharacterUpdateStats, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentCharacterIncreaseIntResponse, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentCharacterIncreaseStrResponse, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentInventoryItemUseResponse, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentInventoryOperationResponse, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityUpdateMoveSpeed, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityRemoveOwnership, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityGroupspawnData, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntitySpawn, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityDespawn, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentSkillLearnResponse, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentSkillMasteryLearnResponse, packetHandleFunction);

  //   Misc. packets
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentActionDeselectResponse, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentActionSelectResponse, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentActionTalkResponse, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentAlchemyElixirResponse, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentAlchemyStoneResponse, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentInventoryRepairResponse, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentInventoryUpdateDurability, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentInventoryUpdateItem, packetHandleFunction);
  // packetBroker_.subscribeToClientPacket(packet::Opcode::kClientAgentActionDeselectRequest, packetHandleFunction);
  // packetBroker_.subscribeToClientPacket(packet::Opcode::kClientAgentActionSelectRequest, packetHandleFunction);
  packetBroker_.subscribeToClientPacket(packet::Opcode::kClientAgentActionTalkRequest, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityUpdatePoints, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityUpdateExperience, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentGuildStorageData, packetHandleFunction);
  packetBroker_.subscribeToClientPacket(packet::Opcode::kClientAgentActionCommandRequest, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentActionCommandResponse, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentSkillBegin, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentSkillEnd, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentBuffAdd, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentBuffLink, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentBuffRemove, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentChatUpdate, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentGameReset, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentResurrectOption, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentOperatorResponse, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentFreePvpUpdateResponse, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentInventoryEquipCountdownStart, packetHandleFunction);
}

void PacketProcessor::handlePacket(const PacketContainer &packet) {
  ZoneScopedN("PacketProcessor::handlePacket");
  {
    std::string_view packetOpcodeString = packet::toString(static_cast<packet::Opcode>(packet.opcode));
    ZoneName(packetOpcodeString.data(), packetOpcodeString.size());
  }
  {
    // Do a quick check to see if any packets are coming while Self is despawned.
    static const absl::flat_hash_set<packet::Opcode> expectedOpcodes = {
      packet::Opcode::kFrameworkMessageIdentify,
      packet::Opcode::kServerGatewayPatchResponse,
      packet::Opcode::kServerGatewayShardListResponse,
      packet::Opcode::kServerGatewayLoginResponse,
      packet::Opcode::kFrameworkMessageIdentify,
      packet::Opcode::kServerAgentAuthResponse,
      packet::Opcode::kServerAgentCharacterData,
      packet::Opcode::kServerAgentCharacterSelectionActionResponse,
      packet::Opcode::kServerAgentCharacterSelectionJoinResponse
    };
    const packet::Opcode thisPacketOpcode = static_cast<packet::Opcode>(packet.opcode);
    if (!expectedOpcodes.contains(thisPacketOpcode)) {
      if (!selfEntity_) {
        LOG(WARNING) << absl::StreamFormat("Received packet %s but do not have self entity", packet::toString(thisPacketOpcode));
      }
    }
  }
  std::unique_ptr<packet::parsing::ParsedPacket> parsedPacket;
  try {
    parsedPacket = packetParser_.parsePacket(packet);
  } catch (std::exception &ex) {
    LOG(ERROR) << "Failed to parse packet " << std::hex << packet.opcode << std::dec << ". \"" << ex.what() << '"';
    return;
  }

  if (!parsedPacket) {
    // Not yet parsing this packet
    LOG(ERROR) << "Subscribed to a packet which we're not yet parsing " << std::hex << packet.opcode << std::dec;
    return;
  }
  std::unique_lock worldStateLock(worldState_.mutex);

  try {
    // Login packet handlers
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::FrameworkMessageIdentify, frameworkMessageIdentifyReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerGatewayPatchResponse, serverGatewayPatchResponseReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerGatewayShardListResponse, serverGatewayShardListResponseReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerGatewayLoginResponse, serverGatewayLoginResponseReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerGatewayLoginIbuvChallenge, serverGatewayLoginIbuvChallengeReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentAuthResponse, serverAgentAuthResponseReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentCharacterSelectionActionResponse, serverAgentCharacterSelectionActionResponseReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentCharacterSelectionJoinResponse, serverAgentCharacterSelectionJoinResponseReceived);

    // Movement packet handlers
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentEntityUpdateAngle, serverAgentEntityUpdateAngleReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentEntityUpdateMovement, serverAgentEntityUpdateMovementReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentEntityUpdatePosition, serverAgentEntityUpdatePositionReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentEntitySyncPosition, serverAgentEntitySyncPositionReceived);

    // Character info packet handlers
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ParsedClientItemMove, clientItemMoveReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentCharacterData, serverAgentCharacterDataReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentCosData, serverAgentCosDataReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ParsedServerAgentInventoryStorageData, serverAgentInventoryStorageDataReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentEntityUpdateHwanLevel, serverAgentEntityUpdateHwanLevelReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentEntityUpdateState, serverAgentEntityUpdateStateReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentEntityUpdateMoveSpeed, serverAgentEntityUpdateMoveSpeedReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentEntityRemoveOwnership, serverAgentEntityRemoveOwnershipReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentEntityUpdateStatus, serverAgentEntityUpdateStatusReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentEntityDamageEffect, serverAgentEntityDamageEffectReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentAbnormalInfo, serverAgentAbnormalInfoReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentCharacterUpdateStats, serverAgentCharacterUpdateStatsReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentCharacterIncreaseIntResponse, serverAgentCharacterIncreaseIntResponseReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentCharacterIncreaseStrResponse, serverAgentCharacterIncreaseStrResponseReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentInventoryItemUseResponse, serverAgentInventoryItemUseResponseReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentInventoryOperationResponse, serverAgentInventoryOperationResponseReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentEntityGroupSpawnData, serverAgentEntityGroupSpawnDataReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentEntitySpawn, serverAgentEntitySpawnReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentEntityDespawn, serverAgentEntityDespawnReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentSkillLearnResponse, serverAgentSkillLearnResponseReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentSkillMasteryLearnResponse, serverAgentSkillMasteryLearnResponseReceived);

    // Misc. packets
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentActionDeselectResponse, serverAgentDeselectResponseReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentActionSelectResponse, serverAgentSelectResponseReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentActionTalkResponse, serverAgentTalkResponseReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentAlchemyElixirResponse, serverAgentAlchemyElixirResponseReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentAlchemyStoneResponse, serverAgentAlchemyStoneResponseReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentInventoryRepairResponse, serverAgentInventoryRepairResponseReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentInventoryUpdateDurability, serverAgentInventoryUpdateDurabilityReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentInventoryUpdateItem, serverAgentInventoryUpdateItemReceived);
    // TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ClientAgentActionDeselectRequest, clientAgentActionDeselectRequestReceived);
    // TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ClientAgentActionSelectRequest, clientAgentActionSelectRequestReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ClientAgentActionTalkRequest, clientAgentActionTalkRequestReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentEntityUpdatePoints, serverAgentEntityUpdatePointsReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentEntityUpdateExperience, serverAgentEntityUpdateExperienceReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentGuildStorageData, serverAgentGuildStorageDataReceived);

    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ClientAgentActionCommandRequest, clientAgentActionCommandRequestReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentActionCommandResponse, serverAgentActionCommandResponseReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentSkillBegin, serverAgentSkillBeginReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentSkillEnd, serverAgentSkillEndReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentBuffAdd, serverAgentBuffAddReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentBuffLink, serverAgentBuffLinkReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentBuffRemove, serverAgentBuffRemoveReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentChatUpdate, serverAgentChatUpdateReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentGameReset, serverAgentGameResetReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentResurrectOption, serverAgentResurrectOptionReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentOperatorResponse, serverAgentOperatorResponseReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentFreePvpUpdateResponse, serverAgentFreePvpUpdateResponseReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentInventoryEquipCountdownStart, serverAgentInventoryEquipCountdownStartReceived);
  } catch (std::exception &ex) {
    LOG(ERROR) << absl::StreamFormat("Error while handling packet %s: \"%s\"", packet::toString(static_cast<packet::Opcode>(packet.opcode)), ex.what());
    return;
  }

  LOG(WARNING) << absl::StreamFormat("Unhandled packet (%s; %#06x) subscribed to.", packet::toString(static_cast<packet::Opcode>(packet.opcode)), packet.opcode);
  return;
}

// ============================================================================================================================
// ===============================================Login process packet handling================================================
// ============================================================================================================================

void PacketProcessor::frameworkMessageIdentifyReceived(const packet::parsing::FrameworkMessageIdentify &packet) const {
  // This packet is a response to the client sending 0x2001 where the client indicates that it is the "SR_Client"
  if (packet.moduleName() == "AgentServer") {
    // Connected to agentserver.
    eventBroker_.publishEvent<event::ConnectedToAgentServer>(sessionId_);
  }
}

void PacketProcessor::serverGatewayPatchResponseReceived(const packet::parsing::ServerGatewayPatchResponse &packet) const {
  VLOG(1) << "Gateway patch response received, result: " << static_cast<int>(packet.result());
  eventBroker_.publishEvent<event::GatewayPatchResponseReceived>(sessionId_);
}

void PacketProcessor::serverGatewayShardListResponseReceived(const packet::parsing::ServerGatewayShardListResponse &packet) {
  VLOG(1) << absl::StreamFormat("Shard list:\n%s", absl::StrJoin(packet.shards(), ",\n", [](std::string *out, const packet::structures::Shard &shard) {
    absl::StrAppend(out, shard.toString());
  }));
  if (!worldState_.shardListResponse_.has_value()) {
    worldState_.shardListResponse_.emplace(packet);
  } else {
    // Shard list received again. Check that it matches what we already have.
    auto validateShards = [](std::vector<packet::structures::Shard> shardsCurrent, std::vector<packet::structures::Shard> shardsNew) {
      if (shardsCurrent.size() != shardsNew.size()) {
        // We got a different number of shards, that is a problem.
        return false;
      }
      // Sort both shard lists on shard ID, in case different packets send packets with shards in different orders.
      std::sort(shardsCurrent.begin(), shardsCurrent.end(), [](const packet::structures::Shard &a, const packet::structures::Shard &b) {
        return a.shardId < b.shardId;
      });
      std::sort(shardsNew.begin(), shardsNew.end(), [](const packet::structures::Shard &a, const packet::structures::Shard &b) {
        return a.shardId < b.shardId;
      });
      // Now, compare shards elementwise.
      for (size_t i = 0; i < shardsCurrent.size(); ++i) {
        const packet::structures::Shard &currentShard = shardsCurrent[i];
        const packet::structures::Shard &newShard = shardsNew[i];
        if (currentShard.shardId != newShard.shardId) {
          // Different shard ids is a problem.
          return false;
        }
        if (currentShard.shardName != newShard.shardName) {
          // Different shard name is a problem.
          return false;
        }
        if (currentShard.capacity != newShard.capacity) {
          // Different capacity is weird, we'll allow it for now.
          LOG(WARNING) << absl::StreamFormat("Weird, we saved shard %s (%d) with capacity %d, but now it has capacity %d", currentShard.shardName, currentShard.shardId, currentShard.capacity, newShard.capacity);
        }
        if (currentShard.isOperating != newShard.isOperating) {
          // Different operating status is a problem.
          // TODO: If it goes from not operating to operating, we should probably be happy about that.
          return false;
        }
        if (currentShard.farmId != newShard.farmId) {
          // Not sure what farmId is, but we'll say that a difference is a problem.
          return false;
        }
      }
      return true;
    };

    const bool shardsAreSame = validateShards(worldState_.shardListResponse_->shards(), packet.shards());

    if (!shardsAreSame) {
      LOG(WARNING) << absl::StreamFormat("Current shard list:\n%s", absl::StrJoin(worldState_.shardListResponse_->shards(), ",\n", [](std::string *out, const packet::structures::Shard &shard) {
        absl::StrAppend(out, shard.toString());
      }));
      LOG(WARNING) << absl::StreamFormat("New shard list:\n%s", absl::StrJoin(packet.shards(), ",\n", [](std::string *out, const packet::structures::Shard &shard) {
        absl::StrAppend(out, shard.toString());
      }));
      throw std::runtime_error("Received a different shard list than what we already have");
    }

    // While we say the shards are the same, that's not entirely true. They might be the same shards, but the current online count might have changed. As long as the shards themselves didnt fundamentally change, we're fine. Overwrite with the new list.
    worldState_.shardListResponse_.emplace(packet);
  }
  eventBroker_.publishEvent<event::ShardListReceived>(sessionId_, packet.shards());
}

void PacketProcessor::serverGatewayLoginResponseReceived(const packet::parsing::ServerGatewayLoginResponse &packet) const {
  // TODO: This data should be sent in an event, rather than stored in the world state.
  if (packet.result() == packet::enums::LoginResult::kSuccess) {
    VLOG(1) << "Received login response with token " << packet.agentServerToken();
    eventBroker_.publishEvent<event::GatewayLoginResponseReceived>(sessionId_, packet.agentServerToken());
  } else {
    // TODO: Send an event.
    LOG(WARNING) << " Login failed";
  }
}

void PacketProcessor::serverGatewayLoginIbuvChallengeReceived(const packet::parsing::ServerGatewayLoginIbuvChallenge &packet) const {
  // Got the captcha prompt, respond with an answer
  VLOG(1) << "Received captcha challenge";
  eventBroker_.publishEvent<event::IbuvChallengeReceived>(sessionId_);
}

void PacketProcessor::serverAgentAuthResponseReceived(const packet::parsing::ServerAgentAuthResponse &packet) const {
  if (packet.result() == 0x01) {
    // Successful login
    eventBroker_.publishEvent<event::ServerAuthSuccess>(sessionId_);
  }
}

void PacketProcessor::serverAgentCharacterSelectionActionResponseReceived(const packet::parsing::ServerAgentCharacterSelectionActionResponse &packet) const {
  if (packet.result() == 1) {
    if (packet.action() == packet::enums::CharacterSelectionAction::kList) {
      VLOG(1) << "Character list received. " << packet.characters().size() << " character(s)";
      eventBroker_.publishEvent<event::CharacterListReceived>(sessionId_, packet.characters());
    } else {
      LOG(INFO) << "Other character selection action success: " << static_cast<int>(packet.action());
    }
  } else {
    LOG(WARNING) << "Error during character selection action " << static_cast<int>(packet.action()) << ", error code: " << packet.errorCode();
  }
}

void PacketProcessor::serverAgentCharacterSelectionJoinResponseReceived(const packet::parsing::ServerAgentCharacterSelectionJoinResponse &packet) const {
  // A character was selected after login, this is the response
  if (packet.result() == 1) {
    eventBroker_.publishEvent<event::CharacterSelectionJoinSuccess>(sessionId_);
  } else {
    // Character selection failed
    // TODO: Properly handle error
    LOG(WARNING) << "Failed when selecting character";
  }
}

// ============================================================================================================================
// ==================================================Movement packet handling==================================================
// ============================================================================================================================

void PacketProcessor::serverAgentEntityUpdateAngleReceived(packet::parsing::ServerAgentEntityUpdateAngle &packet) const {
  std::shared_ptr<entity::MobileEntity> mobileEntity = worldState_.getEntity<entity::MobileEntity>(packet.globalId());
  if (mobileEntity->moving()) {
    if (mobileEntity->destinationPosition) {
      throw std::runtime_error("Got angle update, but we're running to a destination");
    }
    if (mobileEntity->angle() != packet.angle()) {
      // Changed angle while running
      mobileEntity->setMovingTowardAngle(std::nullopt, packet.angle());
    }
  } else {
    mobileEntity->setAngle(packet.angle());
  }
}

void PacketProcessor::serverAgentEntitySyncPositionReceived(packet::parsing::ServerAgentEntitySyncPosition &packet) const {
  std::shared_ptr<entity::MobileEntity> mobileEntity = worldState_.getEntity<entity::MobileEntity>(packet.globalId());
  mobileEntity->syncPosition(packet.position());
}

void PacketProcessor::serverAgentEntityUpdatePositionReceived(packet::parsing::ServerAgentEntityUpdatePosition &packet) const {
  std::shared_ptr<entity::MobileEntity> mobileEntity = worldState_.getEntity<entity::MobileEntity>(packet.globalId());
  mobileEntity->setStationaryAtPosition(packet.position());
}

void PacketProcessor::serverAgentEntityUpdateMovementReceived(packet::parsing::ServerAgentEntityUpdateMovement &packet) const {
  std::shared_ptr<entity::MobileEntity> mobileEntity = worldState_.getEntity<entity::MobileEntity>(packet.globalId());
  std::optional<sro::Position> sourcePosition;
  if (packet.hasSource()) {
    // Server is telling us our source position
    sourcePosition = packet.sourcePosition();
  }
  if (packet.hasDestination()) {
    mobileEntity->setMovingToDestination(sourcePosition, packet.destinationPosition());
  } else {
    mobileEntity->setMovingTowardAngle(sourcePosition, packet.angle());
  }
}

// ============================================================================================================================
// ===============================================Character info packet handling===============================================
// ============================================================================================================================

void PacketProcessor::clientItemMoveReceived(const packet::parsing::ParsedClientItemMove &packet) const {
  const auto itemMovement = packet.movement();
  if (itemMovement.type == packet::enums::ItemMovementType::kBuyItem) {
    // User is buying something from the store
    selfEntity_->setUserPurchaseRequest(itemMovement);
  }
}

void initializeSelfFromCharacterDataPacket(entity::Self &self, const packet::parsing::ServerAgentCharacterData &packet) {
  self.initializeCurrentHp(packet.hp());
  self.initializeCurrentMp(packet.mp());
  self.setHwanLevel(packet.hwanLevel());
  self.initializeCurrentLevel(packet.curLevel());
  self.initializeSkillPoints(packet.skillPoints());
  self.initializeAvailableStatPoints(packet.availableStatPoints());
  self.initializeHwanPoints(packet.hwanPoints());
  self.initializeCurrentExpAndSpExp(packet.currentExperience(), packet.currentSpExperience());
  self.setMasteriesAndSkills(packet.masteries(), packet.skills());

  // Position
  // TODO: Handle the case when the character spawns in a moving state
  self.initializePosition(packet.position());
  // self.setStationaryAtPosition(packet.position());
  self.initializeAngle(packet.angle());

  // State
  self.setLifeState(packet.lifeState());
  self.setMotionState(packet.motionState());
  self.initializeBodyState(packet.bodyState());

  // Buffs
  // TODO: If we spawn with any active buffs, add them
  // worldState_.addBuff(packet.globalId(), packet.skillRefId(), packet.activeBuffToken());

  // Speed
  self.setSpeed(packet.walkSpeed(), packet.runSpeed());
  self.setHwanSpeed(packet.hwanSpeed());
  self.name = packet.characterName();
  self.initializeGold(packet.gold());
  const auto inventorySize = packet.inventorySize();
  const auto &inventoryItemMap = packet.inventoryItemMap();
  helpers::initializeInventory(self.inventory, inventorySize, inventoryItemMap);
  const auto avatarInventorySize = packet.avatarInventorySize();
  const auto &avatarInventoryItemMap = packet.avatarInventoryItemMap();
  helpers::initializeInventory(self.avatarInventory, avatarInventorySize, avatarInventoryItemMap);
}

void PacketProcessor::serverAgentCharacterDataReceived(const packet::parsing::ServerAgentCharacterData &packet) {
  if (ABSL_VLOG_IS_ON(1)) {
    // ================= Masteries =================
    VLOG(1) << "Masteries:";
    for (const auto &m : packet.masteries()) {
      const auto &mastery = gameData_.masteryData().getMasteryById(m.id);
      VLOG(1) << "  Mastery " << mastery.masteryNameCode << "(" << m.id << ") is level " << (int)m.level;
    }
    // ================== Skills ===================
    std::vector<std::pair<std::string, sro::pk2::ref::Skill::Param1Type>> skillTypes = {
      // TODO: These labels are wrong!
      {"Melee skills", sro::pk2::ref::Skill::Param1Type::kMelee},
      {"Ranged skills", sro::pk2::ref::Skill::Param1Type::kRanged},
      {"Buffs", sro::pk2::ref::Skill::Param1Type::kBuff},
      {"Passive skills", sro::pk2::ref::Skill::Param1Type::kPassive},
    };
    for (const auto &i : skillTypes) {
      std::stringstream ss;
      ss << i.first << ": [ ";
      for (const auto &s : packet.skills()) {
        const auto &skillData = gameData_.skillData().getSkillById(s.id);
        if (skillData.param1Type() == i.second) {
          std::optional<std::string> maybeSkillName;
          constexpr const bool kLogName{true};
          if constexpr (kLogName) {
            maybeSkillName = gameData_.getSkillName(s.id);
          }
          // Print skill
          if (maybeSkillName) {
            ss << absl::StreamFormat("{%s:%d}, ", *maybeSkillName, s.id);
          } else {
            ss << absl::StreamFormat("{%d}, ", s.id);
          }
        }
      }
      ss << "]";
      VLOG(1) << ss.str();
    }
    // ================= Inventory =================
    const auto &inventoryItemMap = packet.inventoryItemMap();
    VLOG(1) << "Inventory:";
    for (const auto [slotNum,itemPtr] : inventoryItemMap) {
      if (itemPtr != nullptr) {
        VLOG(1) << "  #" << static_cast<int>(slotNum) << ": " << itemPtr->refItemId << ',' << gameData_.getItemName(itemPtr->refItemId);
      } else {
        VLOG(1) << "  #" << static_cast<int>(slotNum) << ": empty";
      }
    }
  }

  std::shared_ptr<entity::Self> selfEntity = std::make_shared<entity::Self>(gameData_, packet.globalId(), packet.refObjId(), packet.jId());
  initializeSelfFromCharacterDataPacket(*selfEntity.get(), packet);
  VLOG(1) << "GID:" << selfEntity->globalId << ", and we have " << selfEntity->currentHp() << " hp and " << selfEntity->currentMp() << " mp";
  worldState_.entitySpawned(std::move(selfEntity), eventBroker_);
  selfEntity_ = worldState_.entityTracker().getEntity<entity::Self>(packet.globalId());
  LOG(INFO) << "Self entity saved with ID " << selfEntity_->globalId;

  // We send a distinct SelfSpawned event with our session ID so that `Bot` can know who we are.
  // Note: entityTracker().entitySpawned() also sends an EntitySpawned event.
  eventBroker_.publishEvent<event::SelfSpawned>(sessionId_, packet.globalId());
}

void PacketProcessor::serverAgentCosDataReceived(const packet::parsing::ServerAgentCosData &packet) const {
  if (packet.isAbilityPet()) {
    if (selfEntity_ && packet.ownerGlobalId() == selfEntity_->globalId) {
      // Is our pickpet
      auto it = selfEntity_->cosInventoryMap.find(packet.globalId());
      if (it == selfEntity_->cosInventoryMap.end()) {
        // Not yet tracking this Cos
        auto emplaceResult = selfEntity_->cosInventoryMap.emplace(packet.globalId(), storage::Storage());
        if (!emplaceResult.second) {
          throw std::runtime_error("Unable to create new Cos inventory");
        }
        auto &cosInventory = emplaceResult.first->second;
        helpers::initializeInventory(cosInventory, packet.inventorySize(), packet.inventoryItemMap());
        eventBroker_.publishEvent<event::CosSpawned>(packet.globalId());
      } else {
        throw std::runtime_error("Already tracking this Cos");
        // Maybe we should ensure that we never get here
        // On teleport, our COS globalId will change
        // On resummon, our COS globalId will change
      }
    } else {
      LOG(INFO) << "Got Cos data for someone else's Cos";
    }
  } else {
    LOG(INFO) << "Non-ability Cos";
  }
}

void PacketProcessor::serverAgentInventoryStorageDataReceived(const packet::parsing::ParsedServerAgentInventoryStorageData &packet) const {
  selfEntity_->setStorageGold(packet.gold());
  helpers::initializeInventory(selfEntity_->storage, packet.storageSize(), packet.storageItemMap());
  selfEntity_->haveOpenedStorageSinceTeleport = true;
  eventBroker_.publishEvent(event::EventCode::kStorageInitialized);
}

void PacketProcessor::serverAgentEntityUpdateHwanLevelReceived(packet::parsing::ServerAgentEntityUpdateHwanLevel &packet) const {
  if (selfEntity_ && packet.globalId() == selfEntity_->globalId) {
    selfEntity_->setHwanLevel(packet.hwanLevel());
  }
}

void PacketProcessor::serverAgentEntityUpdateStateReceived(packet::parsing::ServerAgentEntityUpdateState &packet) const {
  if (packet.stateType() == packet::enums::StateType::kMotionState) {
    std::shared_ptr<entity::MobileEntity> mobileEntity = worldState_.getEntity<entity::MobileEntity>(packet.globalId());
    mobileEntity->setMotionState(static_cast<entity::MotionState>(packet.state()));
  } else if (packet.stateType() == packet::enums::StateType::kLifeState) {
    std::shared_ptr<entity::Character> characterEntity = worldState_.getEntity<entity::Character>(packet.globalId());
    const auto newLifeState = static_cast<sro::entity::LifeState>(packet.state());
    characterEntity->setLifeState(newLifeState);
  } else if (selfEntity_ && packet.globalId() == selfEntity_->globalId) {
    if (packet.stateType() == packet::enums::StateType::kBodyState) {
      selfEntity_->setBodyState(static_cast<packet::enums::BodyState>(packet.state()));
    }
  }
}

void PacketProcessor::serverAgentEntityUpdateMoveSpeedReceived(const packet::parsing::ServerAgentEntityUpdateMoveSpeed &packet) const {
  std::shared_ptr<entity::MobileEntity> mobileEntity = worldState_.getEntity<entity::MobileEntity>(packet.globalId());
  mobileEntity->setSpeed(packet.walkSpeed(), packet.runSpeed());
}

void PacketProcessor::serverAgentEntityRemoveOwnershipReceived(const packet::parsing::ServerAgentEntityRemoveOwnership &packet) const {
  std::shared_ptr<entity::Item> itemEntity = worldState_.getEntity<entity::Item>(packet.globalId());
  itemEntity->removeOwnership();
}

void PacketProcessor::serverAgentEntityUpdateStatusReceived(const packet::parsing::ServerAgentEntityUpdateStatus &packet) const {
  if (selfEntity_ && packet.globalId() == selfEntity_->globalId) {
    if (flags::isSet(packet.vitalBitmask(), packet::enums::VitalInfoFlag::kVitalInfoHp)) {
      // Our HP changed
      selfEntity_->setCurrentHp(packet.newHpValue());
    }
    if (flags::isSet(packet.vitalBitmask(), packet::enums::VitalInfoFlag::kVitalInfoMp)) {
      // Our MP changed
      selfEntity_->setCurrentMp(packet.newMpValue());
    }

    if (flags::isSet(packet.vitalBitmask(), packet::enums::VitalInfoFlag::kVitalInfoAbnormal)) {
      // Our states changed
      auto stateBitmask = packet.stateBitmask();
      auto stateLevels = packet.stateLevels();
      selfEntity_->updateStates(stateBitmask, stateLevels);
    }
  } else {
    // Not for my character
    if (flags::isSet(packet.vitalBitmask(), packet::enums::VitalInfoFlag::kVitalInfoHp)) {
      std::shared_ptr<entity::Character> character = worldState_.getEntity<entity::Character>(packet.globalId());
      character->setCurrentHp(packet.newHpValue());
    }
  }
}

void PacketProcessor::serverAgentEntityDamageEffectReceived(const packet::parsing::ServerAgentEntityDamageEffect &packet) const {
  std::shared_ptr<entity::Entity> entity = worldState_.getEntity(packet.globalId());
  if (auto *entityAsCharacter = dynamic_cast<entity::Character*>(entity.get())) {
    if (entityAsCharacter->currentHpIsKnown()) {
      entityAsCharacter->setCurrentHp(std::max<int64_t>(0, int64_t(entityAsCharacter->currentHp()) - packet.effectDamage()));
    }
  }
  // This packet only comes for effects which we deal.
  if (selfEntity_) {
    eventBroker_.publishEvent<event::DealtDamage>(selfEntity_->globalId, packet.globalId(), packet.effectDamage());
  }
}

void PacketProcessor::serverAgentAbnormalInfoReceived(const packet::parsing::ServerAgentAbnormalInfo &packet) const {
  for (int i=0; i<=helpers::toBitNum(packet::enums::AbnormalStateFlag::kZombie); ++i) {
    selfEntity_->setLegacyStateEffect(helpers::fromBitNum(i), packet.states().at(i).effectOrLevel);
  }
  eventBroker_.publishEvent<event::EntityStatesChanged>(selfEntity_->globalId);
}

void PacketProcessor::serverAgentCharacterUpdateStatsReceived(const packet::parsing::ServerAgentCharacterUpdateStats &packet) const {
  selfEntity_->setMaxHpMp(packet.maxHp(), packet.maxMp());
  try {
    selfEntity_->setStatPoints(packet.strPoints(), packet.intPoints());
  } catch (const std::exception &ex) {
    // TODO: Known issue.
    LOG(WARNING) << absl::StreamFormat("Incorrect logic in entity::Self::setStatPoints: \"%s\"", ex.what());
  }
}

void PacketProcessor::serverAgentCharacterIncreaseIntResponseReceived(const packet::parsing::ServerAgentCharacterIncreaseIntResponse &packet) const {
  if (packet.result() != 2) {
    // Success!
    // Note: We could update our int points here, but we instead do it from ServerAgentCharacterUpdateStats, which comes earlier.
  } else {
    if (packet.errorCode() == 29702) {
      LOG(INFO) << "Not enough stat points when trying to increase Int";
    } else {
      LOG(INFO) << "Unknown error when increasing Int points: " << packet.errorCode();
    }
  }
}

void PacketProcessor::serverAgentCharacterIncreaseStrResponseReceived(const packet::parsing::ServerAgentCharacterIncreaseStrResponse &packet) const {
  if (packet.result() != 2) {
    // Success!
    // Note: We could update our str points here, but we instead do it from ServerAgentCharacterUpdateStats, which comes earlier.
  } else {
    if (packet.errorCode() == 29702) {
      LOG(INFO) << "Not enough stat points when trying to increase Str";
    } else {
      LOG(INFO) << "Unknown error when increasing Str points: " << packet.errorCode();
    }
  }
}

void PacketProcessor::serverAgentInventoryItemUseResponseReceived(const packet::parsing::ServerAgentInventoryItemUseResponse &packet) const {
  if (packet.result() != 1) {
    // Failed to use item
    if (packet.errorCode() != packet::enums::InventoryErrorCode::kWaitForReuseDelay &&
        packet.errorCode() != packet::enums::InventoryErrorCode::kCharacterDead &&
        packet.errorCode() != packet::enums::InventoryErrorCode::kItemDoesNotExist) {
      LOG(INFO) << "Unknown error while trying to use an item: " << static_cast<int>(packet.errorCode());
    }
    eventBroker_.publishEvent<event::ItemUseFailed>(selfEntity_->globalId, packet.slotNum(), packet.typeData(), packet.errorCode());
    return;
  }
  // Successfully used an item
  // Make sure we have the item
  if (!selfEntity_->inventory.hasItem(packet.slotNum())) {
    throw std::runtime_error("Used an item, but it's not in our inventory");
  }

  storage::Item *itemPtr = selfEntity_->inventory.getItem(packet.slotNum());
  if (itemPtr == nullptr) {
    throw std::runtime_error("Used an item, but there is not item in this slot of our inventory");
  }
  // Lets double check its type data
  if (packet.typeData() != itemPtr->typeId()) {
    throw std::runtime_error("Used an item, but the stored typeId doesn't match what came in the packet");
  }

  auto *expendableItemPtr = dynamic_cast<storage::ItemExpendable*>(itemPtr);
  // Make sure that the item is an expendable.
  if (expendableItemPtr != nullptr) {
    // Update the item's quantity.
    expendableItemPtr->quantity = packet.remainingCount();
    // Figure out the cooldown of this item.
    const auto itemCooldown = getItemCooldownMs(*expendableItemPtr);
    selfEntity_->usedAnItem(packet.typeData(), itemCooldown);

    // TODO: It feels a bit weird to have redundant information across the following 2 events.
    eventBroker_.publishEvent<event::ItemUseSuccess>(selfEntity_->globalId, packet.slotNum(), itemPtr->refItemId);
    eventBroker_.publishEvent<event::ItemMoved>(selfEntity_->globalId, sro::storage::Position(sro::storage::Storage::kInventory, packet.slotNum()), std::nullopt);
  } else if (dynamic_cast<const storage::ItemCosAbilitySummoner*>(itemPtr) == nullptr &&
              dynamic_cast<const storage::ItemCosGrowthSummoner*>(itemPtr) == nullptr) {
    throw std::runtime_error("Used an item, but it wasn't an expendable or COS summon");
  }
}

std::optional<std::chrono::milliseconds> PacketProcessor::getItemCooldownMs(const storage::ItemExpendable &item) const {
  std::optional<std::chrono::milliseconds> cooldownMilliseconds;
  const auto typeData = item.typeId();
  if (type_id::categories::kRecoveryPotion.contains(typeData)) {
    // Is a potion or grain
    const auto &itemData = gameData_.itemData().getItemById(item.refItemId);
    // param1 = Hp heal amount
    // param2 = Hp heal percent
    // param3 = Mp heal amount
    // param4 = Mp heal percent
    const bool isAGrain = (itemData.param2 > 0 || itemData.param4 > 0);
    if (isAGrain)  {
      if (type_id::categories::kHpPotion.contains(typeData)) {
        cooldownMilliseconds.emplace(selfEntity_->getHpGrainDelay());
      } else if (type_id::categories::kMpPotion.contains(typeData)) {
        cooldownMilliseconds.emplace(selfEntity_->getMpGrainDelay());
      } else if (type_id::categories::kVigorPotion.contains(typeData)) {
        cooldownMilliseconds.emplace(selfEntity_->getVigorGrainDelay());
      }
    } else {
      if (type_id::categories::kHpPotion.contains(typeData)) {
        cooldownMilliseconds.emplace(selfEntity_->getHpPotionDelay());
      } else if (type_id::categories::kMpPotion.contains(typeData)) {
        cooldownMilliseconds.emplace(selfEntity_->getMpPotionDelay());
      } else if (type_id::categories::kVigorPotion.contains(typeData)) {
        cooldownMilliseconds.emplace(selfEntity_->getVigorPotionDelay());
      }
    }
  } else if (type_id::categories::kUniversalPill.contains(typeData)) {
    cooldownMilliseconds.emplace(selfEntity_->getUniversalPillDelay());
  } else if (type_id::categories::kPurificationPill.contains(typeData)) {
    cooldownMilliseconds.emplace(selfEntity_->getPurificationPillDelay());
  } else if (type_id::categories::kRepair.contains(typeData)) {
    // Item mall repair hammer is hard-coded in the server binary with a cooldown of 1000ms.
    cooldownMilliseconds.emplace(1000);
  }
  return cooldownMilliseconds;
}

void PacketProcessor::serverAgentInventoryOperationResponseReceived(const packet::parsing::ServerAgentInventoryOperationResponse &packet) const {
  auto addItemToInventory = [this](storage::Storage &inventory, const std::shared_ptr<storage::Item> &newItem, const sro::scalar_types::StorageIndexType destSlot) {
    if (newItem != nullptr) {
      // Picked up an item
      if (inventory.hasItem(destSlot)) {
        // There is already something in this slot
        auto existingItem = inventory.getItem(destSlot);
        bool addedToStack = false;
        if (existingItem->refItemId == newItem->refItemId) {
          // Both items have the same refId
          storage::ItemExpendable *newExpendableItem, *existingExpendableItem;
          if ((newExpendableItem = dynamic_cast<storage::ItemExpendable*>(newItem.get())) &&
              (existingExpendableItem = dynamic_cast<storage::ItemExpendable*>(existingItem))) {
            // Both items are expendables, so we can stack them
            // Picked item's quantity (if an expendable) is the total in the given slot
            existingExpendableItem->quantity = newExpendableItem->quantity;
            addedToStack = true;
          }
        }
        if (!addedToStack) {
          throw std::runtime_error("Item couldn't be added to the stack");
        }
      } else {
        // This is a new item
        inventory.addItem(destSlot, newItem);
        if (!inventory.hasItem(destSlot)) {
          // This is especially weird since we already know that there was nothing in this slot
          throw std::runtime_error("Could not add item to inventory");
        }
      }
    } else {
      throw std::runtime_error("Adding an item to inventory, but the new item is null");
    }
  };

  auto removeItemFromInventory = [&, this](const sro::scalar_types::StorageIndexType slotIndex) {
    if (selfEntity_->inventory.hasItem(slotIndex)) {
      selfEntity_->inventory.deleteItem(slotIndex);
    } else {
      throw std::runtime_error(absl::StrFormat("RemoveItemFromInventory(): There's no item in inventory slot %d", slotIndex));
    }
  };

  // Check if operation failed.
  if (!packet.success()) {
    // Publish an event for the failed operation
    eventBroker_.publishEvent<event::ItemMoveFailed>(selfEntity_->globalId, packet.errorCode());
    return;
  }

  // TODO: If we used an item and it moved, we'll need to update the "reference" to this item in the used item queue
  const std::vector<packet::structures::ItemMovement> &itemMovements = packet.itemMovements();
  for (const auto &movement : itemMovements) {
    if (movement.type == packet::enums::ItemMovementType::kUpdateSlotsInventory) {
      selfEntity_->inventory.moveItem(movement.srcSlot, movement.destSlot, movement.quantity);
      eventBroker_.publishEvent<event::ItemMoved>(selfEntity_->globalId,
                                                  sro::storage::Position(sro::storage::Storage::kInventory, movement.srcSlot),
                                                  sro::storage::Position(sro::storage::Storage::kInventory, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kUpdateSlotsChest) {
      selfEntity_->storage.moveItem(movement.srcSlot, movement.destSlot, movement.quantity);
      eventBroker_.publishEvent<event::ItemMoved>(selfEntity_->globalId,
                                                  sro::storage::Position(sro::storage::Storage::kStorage, movement.srcSlot),
                                                  sro::storage::Position(sro::storage::Storage::kStorage, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kUpdateSlotsGuildChest) {
      selfEntity_->guildStorage.moveItem(movement.srcSlot, movement.destSlot, movement.quantity);
      eventBroker_.publishEvent<event::ItemMoved>(selfEntity_->globalId,
                                                  sro::storage::Position(sro::storage::Storage::kGuildStorage, movement.srcSlot),
                                                  sro::storage::Position(sro::storage::Storage::kGuildStorage, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kChestDepositItem) {
      selfEntity_->storage.addItem(movement.destSlot, selfEntity_->inventory.withdrawItem(movement.srcSlot));
      eventBroker_.publishEvent<event::ItemMoved>(selfEntity_->globalId,
                                                  sro::storage::Position(sro::storage::Storage::kInventory, movement.srcSlot),
                                                  sro::storage::Position(sro::storage::Storage::kStorage, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kChestWithdrawItem) {
      selfEntity_->inventory.addItem(movement.destSlot, selfEntity_->storage.withdrawItem(movement.srcSlot));
      eventBroker_.publishEvent<event::ItemMoved>(selfEntity_->globalId,
                                                  sro::storage::Position(sro::storage::Storage::kStorage, movement.srcSlot),
                                                  sro::storage::Position(sro::storage::Storage::kInventory, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kGuildChestDepositItem) {
      selfEntity_->guildStorage.addItem(movement.destSlot, selfEntity_->inventory.withdrawItem(movement.srcSlot));
      eventBroker_.publishEvent<event::ItemMoved>(selfEntity_->globalId,
                                                  sro::storage::Position(sro::storage::Storage::kInventory, movement.srcSlot),
                                                  sro::storage::Position(sro::storage::Storage::kGuildStorage, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kGuildChestWithdrawItem) {
      selfEntity_->inventory.addItem(movement.destSlot, selfEntity_->guildStorage.withdrawItem(movement.srcSlot));
      eventBroker_.publishEvent<event::ItemMoved>(selfEntity_->globalId,
                                                  sro::storage::Position(sro::storage::Storage::kGuildStorage, movement.srcSlot),
                                                  sro::storage::Position(sro::storage::Storage::kInventory, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kBuyItem) {
      if (selfEntity_->haveUserPurchaseRequest()) {
        const auto userPurchaseRequest = selfEntity_->getUserPurchaseRequest();
        // User purchased something, we saved this so that we can get the NPC's global Id
        if (worldState_.entityTracker().trackingEntity(userPurchaseRequest.globalId)) {
          std::shared_ptr<entity::Entity> entity = worldState_.getEntity(userPurchaseRequest.globalId);
          // Found the NPC which this purchase was made with
          if (gameData_.characterData().haveCharacterWithId(entity->refObjId)) {
            auto npcName = gameData_.characterData().getCharacterById(entity->refObjId).codeName128;
            auto itemInfo = gameData_.shopData().getItemFromNpc(npcName, userPurchaseRequest.storeTabNumber, userPurchaseRequest.storeSlotNumber);
            const auto &itemRef = gameData_.itemData().getItemByCodeName128(itemInfo.refItemCodeName);
            if (movement.destSlots.size() == 1) {
              // Just a single item or single stack
              auto item = helpers::createItemFromScrap(itemInfo, itemRef);
              storage::ItemExpendable *itemExp = dynamic_cast<storage::ItemExpendable*>(item.get());
              if (itemExp != nullptr) {
                itemExp->quantity = movement.quantity;
              }
              selfEntity_->inventory.addItem(movement.destSlots[0], item);
              eventBroker_.publishEvent<event::ItemMoved>(selfEntity_->globalId,
                                                          std::nullopt,
                                                          sro::storage::Position(sro::storage::Storage::kInventory, movement.destSlot));
            } else {
              // Multiple destination slots, must be unstackable items like equipment
              for (auto destSlot : movement.destSlots) {
                auto item = helpers::createItemFromScrap(itemInfo, itemRef);
                selfEntity_->inventory.addItem(destSlot, item);
                eventBroker_.publishEvent<event::ItemMoved>(selfEntity_->globalId,
                                                            std::nullopt,
                                                            sro::storage::Position(sro::storage::Storage::kInventory, movement.destSlot));
              }
            }
          }
        }
        selfEntity_->resetUserPurchaseRequest();
      } else {
        LOG(INFO) << "kBuyItem but we dont have the data from the client packet";
        // TODO: Introduce unknown item concept?
      }
    } else if (movement.type == packet::enums::ItemMovementType::kSellItem) {
      if (selfEntity_->inventory.hasItem(movement.srcSlot)) {
        bool soldEntireStack = true;
        auto item = selfEntity_->inventory.getItem(movement.srcSlot);
        storage::ItemExpendable *itemExpendable;
        if ((itemExpendable = dynamic_cast<storage::ItemExpendable*>(item)) != nullptr) {
          if (itemExpendable->quantity != movement.quantity) {
            LOG(INFO) << "Sold only some of this item " << itemExpendable->quantity << " -> " << itemExpendable->quantity-movement.quantity;
            soldEntireStack = false;
            itemExpendable->quantity -= movement.quantity;
            auto clonedItem = storage::cloneItem(item);
            dynamic_cast<storage::ItemExpendable*>(clonedItem.get())->quantity = movement.quantity;
            selfEntity_->buybackQueue.addItem(clonedItem);
          }
        }
        if (soldEntireStack) {
          auto item = selfEntity_->inventory.withdrawItem(movement.srcSlot);
          selfEntity_->buybackQueue.addItem(item);
        }
        eventBroker_.publishEvent<event::ItemMoved>(selfEntity_->globalId,
                                                    sro::storage::Position(sro::storage::Storage::kInventory, movement.srcSlot),
                                                    std::nullopt);
      } else {
        LOG(INFO) << "Sold an item from a slot that we didn't have item data for";
      }
    } else if (movement.type == packet::enums::ItemMovementType::kBuyback) {
      if (selfEntity_->buybackQueue.hasItem(movement.srcSlot)) {
        if (!selfEntity_->inventory.hasItem(movement.destSlot)) {
          const auto itemPtr = selfEntity_->buybackQueue.getItem(movement.srcSlot);
          // TODO: Track gold change
          //  The amount of gold that this item costs to buyback seems to be equal to the amount that it was sold for
          bool boughtBackAll = true;
          if (movement.quantity > 1) {
            storage::ItemExpendable *itemExpendable = dynamic_cast<storage::ItemExpendable*>(itemPtr);
            if (itemExpendable != nullptr) {
              if (itemExpendable->quantity > movement.quantity) {
                LOG(INFO) << "Only buying back a partial amount from the buyback slot. Didn't know this was possible (" << movement.quantity << '/' << itemExpendable->quantity << ")";
                boughtBackAll = false;
                auto clonedItem = storage::cloneItem(itemPtr);
                itemExpendable->quantity -= movement.quantity;
                dynamic_cast<storage::ItemExpendable*>(clonedItem.get())->quantity = movement.quantity;
                selfEntity_->inventory.addItem(movement.destSlot, clonedItem);
                eventBroker_.publishEvent<event::ItemMoved>(selfEntity_->globalId,
                                                            std::nullopt,
                                                            sro::storage::Position(sro::storage::Storage::kInventory, movement.destSlot));
              }
            }
          }
          if (boughtBackAll) {
            selfEntity_->inventory.addItem(movement.destSlot, selfEntity_->buybackQueue.withdrawItem(movement.srcSlot));
            eventBroker_.publishEvent<event::ItemMoved>(selfEntity_->globalId,
                                                        std::nullopt,
                                                        sro::storage::Position(sro::storage::Storage::kInventory, movement.destSlot));
          }
        } else {
          LOG(INFO) << "Bought back item is being moved into a slot that's already occupied";
        }
      } else {
        LOG(INFO) << "Bought back an item that we weren't tracking";
      }
    } else if (movement.type == packet::enums::ItemMovementType::kPickItem) {
      if (movement.destSlot != packet::structures::ItemMovement::kGoldSlot) {
        addItemToInventory(selfEntity_->inventory, movement.newItem, movement.destSlot);
        eventBroker_.publishEvent<event::ItemMoved>(selfEntity_->globalId,
                                                    std::nullopt,
                                                    sro::storage::Position(sro::storage::Storage::kInventory, movement.destSlot));
      }
    } else if (movement.type == packet::enums::ItemMovementType::kDropItem) {
      removeItemFromInventory(movement.srcSlot);
      eventBroker_.publishEvent<event::ItemMoved>(selfEntity_->globalId,
                                                  sro::storage::Position(sro::storage::Storage::kInventory, movement.srcSlot),
                                                  std::nullopt);
    } else if (movement.type == packet::enums::ItemMovementType::kAddItemByServer) {
      addItemToInventory(selfEntity_->inventory, movement.newItem, movement.destSlot);
      eventBroker_.publishEvent<event::ItemMoved>(selfEntity_->globalId,
                                                  std::nullopt,
                                                  sro::storage::Position(sro::storage::Storage::kInventory, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kRemoveItemByServer) {
      removeItemFromInventory(movement.srcSlot);
      eventBroker_.publishEvent<event::ItemMoved>(selfEntity_->globalId,
                                                  sro::storage::Position(sro::storage::Storage::kInventory, movement.srcSlot),
                                                  std::nullopt);
    } else if (movement.type == packet::enums::ItemMovementType::kDropGold) {
      // Another packet, ServerAgentEntityUpdatePoints, contains character gold update information
    } else if (movement.type == packet::enums::ItemMovementType::kChestWithdrawGold) {
      selfEntity_->setStorageGold(selfEntity_->getStorageGold() - movement.goldAmount);
    } else if (movement.type == packet::enums::ItemMovementType::kChestDepositGold) {
      selfEntity_->setStorageGold(selfEntity_->getStorageGold() + movement.goldAmount);
    } else if (movement.type == packet::enums::ItemMovementType::kGuildChestDepositGold) {
      selfEntity_->setGuildStorageGold(selfEntity_->getGuildStorageGold() - movement.goldAmount);
    } else if (movement.type == packet::enums::ItemMovementType::kGuildChestWithdrawGold) {
      selfEntity_->setGuildStorageGold(selfEntity_->getGuildStorageGold() + movement.goldAmount);
    } else if (movement.type == packet::enums::ItemMovementType::kMoveItemAvatarToInventory) {
      selfEntity_->inventory.addItem(movement.destSlot, selfEntity_->avatarInventory.withdrawItem(movement.srcSlot));
      eventBroker_.publishEvent<event::ItemMoved>(selfEntity_->globalId,
                                                  sro::storage::Position(sro::storage::Storage::kAvatarInventory, movement.srcSlot),
                                                  sro::storage::Position(sro::storage::Storage::kInventory, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kMoveItemInventoryToAvatar) {
      selfEntity_->avatarInventory.addItem(movement.destSlot, selfEntity_->inventory.withdrawItem(movement.srcSlot));
      eventBroker_.publishEvent<event::ItemMoved>(selfEntity_->globalId,
                                                  sro::storage::Position(sro::storage::Storage::kInventory, movement.srcSlot),
                                                  sro::storage::Position(sro::storage::Storage::kAvatarInventory, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kMoveItemCosToInventory) {
      auto &cosInventory = selfEntity_->getCosInventory(movement.globalId);
      selfEntity_->inventory.addItem(movement.destSlot, cosInventory.withdrawItem(movement.srcSlot));
      eventBroker_.publishEvent<event::ItemMoved>(selfEntity_->globalId,
                                                  sro::storage::Position(sro::storage::Storage::kCosInventory, movement.srcSlot),
                                                  sro::storage::Position(sro::storage::Storage::kInventory, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kMoveItemInventoryToCos) {
      auto &cosInventory = selfEntity_->getCosInventory(movement.globalId);
      cosInventory.addItem(movement.destSlot, selfEntity_->inventory.withdrawItem(movement.srcSlot));
      eventBroker_.publishEvent<event::ItemMoved>(selfEntity_->globalId,
                                                  sro::storage::Position(sro::storage::Storage::kInventory, movement.srcSlot),
                                                  sro::storage::Position(sro::storage::Storage::kCosInventory, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kUpdateSlotsInventoryCos) {
      auto &cosInventory = selfEntity_->getCosInventory(movement.globalId);
      cosInventory.moveItem(movement.srcSlot, movement.destSlot, movement.quantity);
      eventBroker_.publishEvent<event::ItemMoved>(selfEntity_->globalId,
                                                  sro::storage::Position(sro::storage::Storage::kCosInventory, movement.srcSlot),
                                                  sro::storage::Position(sro::storage::Storage::kCosInventory, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kPickItemCos) {
      auto &cosInventory = selfEntity_->getCosInventory(movement.globalId);
      addItemToInventory(cosInventory, movement.newItem, movement.destSlot);
      eventBroker_.publishEvent<event::ItemMoved>(selfEntity_->globalId,
                                                  std::nullopt,
                                                  sro::storage::Position(sro::storage::Storage::kCosInventory, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kPickItemByOther) {
      // Always is our COS picking gold. Gold update packet updates our state. We dont need to handle this
    } else {
      LOG(INFO) << "Unknown item movement type: " << static_cast<int>(movement.type);
    }
  }
}

void PacketProcessor::serverAgentEntityGroupSpawnDataReceived(const packet::parsing::ServerAgentEntityGroupSpawnData &packet) const {
  if (packet.groupSpawnType() == packet::enums::GroupSpawnType::kSpawn) {
    for (const std::shared_ptr<entity::Entity> &entity : packet.entities()) {
      if (entity) {
        entitySpawned(entity);
      } else {
        LOG(INFO) << "Received null entity from group spawn";
      }
    }
  } else {
    for (auto globalId : packet.despawnGlobalIds()) {
      entityDespawned(globalId);
    }
  }
}

void PacketProcessor::serverAgentEntitySpawnReceived(const packet::parsing::ServerAgentEntitySpawn &packet) const {
  if (packet.entity()) {
    entitySpawned(std::move(packet.entity()));
  } else {
    LOG(INFO) << "Received null entity from spawn";
  }
}

void PacketProcessor::serverAgentEntityDespawnReceived(const packet::parsing::ServerAgentEntityDespawn &packet) const {
  entityDespawned(packet.globalId());
}

void PacketProcessor::entitySpawned(std::shared_ptr<entity::Entity> entity) const {
  const sro::scalar_types::EntityGlobalId entityGlobalId = entity->globalId;
  const bool firstTimeSeeingEntity = worldState_.entitySpawned(std::move(entity), eventBroker_);
  if (!firstTimeSeeingEntity) {
    // TODO: Does it make sense to execute the below code if the entity is already being tracked?
    // Already aware of this entity, not going to do anything else.
    return;
  }
  // * ~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~ *
  // *  After this point, do not use `entity` anymore.   *
  // *  If the same entity was already being tracked,    *
  // *  we should instead use the entity that is held    *
  // *  in the entity tracker.                           *
  // * ~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~ *

  // Check if the entity spawned in as already moving.
  std::shared_ptr<entity::MobileEntity> mobileEntity = worldState_.entityTracker().getEntity<entity::MobileEntity>(entityGlobalId);
  if (!mobileEntity) {
    // Non-mobile, nothing to do
    return;
  }
  if (mobileEntity->moving()) {
    LOG(INFO) << "Entity is moving, making some changes";
    if (mobileEntity->destinationPosition) {
      // Entity spawned and is moving to a destination
      mobileEntity->setMovingToDestination(mobileEntity->position(), *mobileEntity->destinationPosition);
    } else {
      mobileEntity->setMovingTowardAngle(mobileEntity->position(), mobileEntity->angle());
    }
  }
}

void PacketProcessor::entityDespawned(sro::scalar_types::EntityGlobalId globalId) const {
  if (!worldState_.entityTracker().trackingEntity(globalId)) {
    // TODO: Once eventzones are handled, this check can be removed;
    //  getEntity will throw
    LOG(WARNING) << "Entity despawned, but we're not tracking it";
    return;
  }
  // Destroy entity
  worldState_.entityDespawned(globalId, eventBroker_);
}

void PacketProcessor::serverAgentSkillLearnResponseReceived(const packet::parsing::ServerAgentSkillLearnResponse &packet) const {
  if (packet.success()) {
    selfEntity_->learnSkill(packet.skillId());
  } else {
    LOG(INFO) << "Error learning skill. Error: " << packet.errorCode();
    eventBroker_.publishEvent(event::EventCode::kLearnSkillError);
  }
}

void PacketProcessor::serverAgentSkillMasteryLearnResponseReceived(const packet::parsing::ServerAgentSkillMasteryLearnResponse &packet) const {
  if (packet.success()) {
    selfEntity_->learnMastery(packet.masteryId(), packet.masteryLevel());
  } else {
    LOG(WARNING) << "Error learning mastery. Error: " << packet.errorCode();
  }
}

// ============================================================================================================================
// ============================================================Misc============================================================
// ============================================================================================================================

void PacketProcessor::serverAgentDeselectResponseReceived(const packet::parsing::ServerAgentActionDeselectResponse &packet) const {
  if (packet.result() == 1) {
    // Successfully deselected
    // If there is a talk dialog, and we have an npc selected, it will take 2 deselects to close both dialogs
    //  First, the talk dialog is closed
    if (selfEntity_->talkingGidAndOption) {
      // This closes the talk dialog
      selfEntity_->talkingGidAndOption.reset();
      eventBroker_.publishEvent(event::EventCode::kEntityDeselected);
    } else {
      //  The entity is deselected
      if (selfEntity_->selectedEntity) {
        selfEntity_->selectedEntity.reset();
        eventBroker_.publishEvent(event::EventCode::kEntityDeselected);
      } else {
        LOG(INFO) << "Weird, we didn't have anything selected";
      }
    }
  } else {
    LOG(INFO) << "Deselection failed";
  }
}

void PacketProcessor::serverAgentSelectResponseReceived(const packet::parsing::ServerAgentActionSelectResponse &packet) const {
  if (packet.result() != 1) {
    LOG(INFO) << "Selection failed";
    return;
  }

  // Successfully selected
  // It is possible that we already have something selected. We will just overwrite it
  selfEntity_->selectedEntity = packet.globalId();
  std::shared_ptr<entity::Entity> entity = worldState_.getEntity(packet.globalId());
  if (auto *monster = dynamic_cast<entity::Monster*>(entity.get())) {
    // Selected a monster
    if (flags::isSet(packet.vitalInfoMask(), packet::enums::VitalInfoFlag::kVitalInfoHp)) {
      // Received monster's current HP
      monster->setCurrentHp(packet.hp());
    }
  }
  eventBroker_.publishEvent(event::EventCode::kEntitySelected);
}

void PacketProcessor::serverAgentTalkResponseReceived(const packet::parsing::ServerAgentActionTalkResponse &packet) const {
  if (packet.result() == 1) {
    // Successfully talking to an npc
    if (selfEntity_->pendingTalkGid) {
      // We were waiting for this response
      selfEntity_->talkingGidAndOption = std::make_pair(*selfEntity_->pendingTalkGid, packet.talkOption());
      selfEntity_->pendingTalkGid.reset();
      eventBroker_.publishEvent(event::EventCode::kNpcTalkStart);
    } else {
      LOG(INFO) << "Weird, we weren't expecting to be talking to anything. As a result, we dont know what we're talking to";
    }
  } else {
    LOG(INFO) << "Failed to talk to NPC";
  }
}

void PacketProcessor::serverAgentAlchemyElixirResponseReceived(const packet::parsing::ServerAgentAlchemyElixirResponse &packet) const {
  if (packet.result() == 1) {
    if (!packet.itemWasDestroyed()) {
      // TODO: Improve the inventory API to allow overwriting items
      selfEntity_->inventory.deleteItem(packet.slot());
      selfEntity_->inventory.addItem(packet.slot(), packet.item());
    } else {
      // If the item is destroyed, a server delete packet will remove the item from the inventory.
      LOG(INFO) << "Item was destroyed!";
    }
  }
  eventBroker_.publishEvent(event::EventCode::kAlchemyCompleted);
}

void PacketProcessor::serverAgentAlchemyStoneResponseReceived(const packet::parsing::ServerAgentAlchemyStoneResponse &packet) const {
  if (packet.result() == 1) {
    // TODO: Improve the inventory API to allow overwriting items
    selfEntity_->inventory.deleteItem(packet.slot());
    selfEntity_->inventory.addItem(packet.slot(), packet.item());
  }
  eventBroker_.publishEvent(event::EventCode::kAlchemyCompleted);
}

void PacketProcessor::serverAgentInventoryRepairResponseReceived(const packet::parsing::ServerAgentInventoryRepairResponse &packet) const {
  if (packet.successful()) {
    eventBroker_.publishEvent(event::EventCode::kRepairSuccessful);
  } else {
    LOG(INFO) << "Repairing item(s) failed! Error code: " << packet.errorCode();
  }
}

void PacketProcessor::serverAgentInventoryUpdateDurabilityReceived(const packet::parsing::ServerAgentInventoryUpdateDurability &packet) const {
  if (!selfEntity_->inventory.hasItem(packet.slotIndex())) {
    throw std::runtime_error("Received durability update for inventory slot where no item exists");
  }
  auto *item = selfEntity_->inventory.getItem(packet.slotIndex());
  if (item == nullptr) {
    throw std::runtime_error("Received durability update for inventory item which is null");
  }
  auto *itemAsEquip = dynamic_cast<storage::ItemEquipment*>(item);
  if (itemAsEquip == nullptr) {
    throw std::runtime_error("Received durability update for inventory item which is not a piece of equipment");
  }
  // Update item's durability
  itemAsEquip->durability = packet.durability();
  eventBroker_.publishEvent<event::InventoryItemUpdated>(selfEntity_->globalId, sro::storage::Position(sro::storage::Storage::kInventory, packet.slotIndex()));
}

void PacketProcessor::serverAgentInventoryUpdateItemReceived(const packet::parsing::ServerAgentInventoryUpdateItem &packet) const {
  if (!selfEntity_->inventory.hasItem(packet.slotIndex())) {
    throw std::runtime_error("Received item update for inventory slot where no item exists");
  }
  auto *item = selfEntity_->inventory.getItem(packet.slotIndex());
  if (item == nullptr) {
    throw std::runtime_error("Received item update for inventory item which is null");
  }
  if (packet.itemUpdateHasFlag(packet::enums::ItemUpdateFlag::kQuantity)) {
    // Known reasons for this update: alchemy
    // Try to cast item as expendable
    if (auto *itemAsExpendable = dynamic_cast<storage::ItemExpendable*>(item)) {
      const bool increased = (packet.quantity() > itemAsExpendable->quantity);
      itemAsExpendable->quantity = packet.quantity();
      if (increased) {
        eventBroker_.publishEvent<event::ItemMoved>(selfEntity_->globalId,
                                                    std::nullopt,
                                                    sro::storage::Position(sro::storage::Storage::kInventory, packet.slotIndex()));
      } else {
        eventBroker_.publishEvent<event::ItemMoved>(selfEntity_->globalId,
                                                    sro::storage::Position(sro::storage::Storage::kInventory, packet.slotIndex()),
                                                    std::nullopt);
      }
    } else {
      throw std::runtime_error("Item quantity updated, but this item is not an expendable");
    }
  }
}

// void PacketProcessor::clientAgentActionDeselectRequestReceived(const packet::parsing::ClientAgentActionDeselectRequest &packet) const {
// }

// void PacketProcessor::clientAgentActionSelectRequestReceived(const packet::parsing::ClientAgentActionSelectRequest &packet) const {
// }

void PacketProcessor::clientAgentActionTalkRequestReceived(const packet::parsing::ClientAgentActionTalkRequest &packet) const {
  if (selfEntity_->pendingTalkGid) {
    LOG(INFO) << "Weird, we're already waiting on a response from the server to talk to someone";
  } else {
    selfEntity_->pendingTalkGid = packet.gId();
  }
}

void PacketProcessor::serverAgentEntityUpdatePointsReceived(const packet::parsing::ServerAgentEntityUpdatePoints &packet) const {
  if (packet.updatePointsType() == packet::enums::UpdatePointsType::kGold) {
    selfEntity_->setGold(packet.gold());
  } else if (packet.updatePointsType() == packet::enums::UpdatePointsType::kSp) {
    selfEntity_->setSkillPoints(packet.skillPoints());
  } else if (packet.updatePointsType() == packet::enums::UpdatePointsType::kHwan) {
    selfEntity_->setHwanPoints(packet.hwanPoints());
  } else if (packet.updatePointsType() == packet::enums::UpdatePointsType::kStatPoint) {
    selfEntity_->setAvailableStatPoints(packet.statPoints());
  }
}

void PacketProcessor::serverAgentEntityUpdateExperienceReceived(const packet::parsing::ServerAgentEntityUpdateExperience &packet) const {
  const constexpr int kSpExperienceRequired{400}; // TODO: Move to a central location
  auto currentLevel = selfEntity_->getCurrentLevel();
  const auto levelBefore = currentLevel;
  auto maxExpOfCurrentLevel = gameData_.levelData().getLevel(currentLevel).exp_C;
  int64_t newExperience = selfEntity_->getCurrentExperience() + packet.gainedExperiencePoints();
  if (packet.gainedExperiencePoints() > 0) {
    // Maybe we gained enough experience to level up.
    while (newExperience >= maxExpOfCurrentLevel) {
      newExperience -= maxExpOfCurrentLevel;
      ++currentLevel;
      maxExpOfCurrentLevel = gameData_.levelData().getLevel(currentLevel).exp_C;
    }
  } else if (packet.gainedExperiencePoints() < 0) {
    // Maybe we lost enough experience to level down.
    while (newExperience < 0) {
      --currentLevel;
      maxExpOfCurrentLevel = gameData_.levelData().getLevel(currentLevel).exp_C;
      newExperience += maxExpOfCurrentLevel;
    }
  }
  const uint64_t newSpExperience = (selfEntity_->getCurrentSpExperience() + packet.gainedSpExperiencePoints()) % kSpExperienceRequired;
  selfEntity_->setCurrentExpAndSpExp(newExperience, newSpExperience);
  if (currentLevel != levelBefore) {
    // Our level changed!
    selfEntity_->setCurrentLevel(currentLevel);
  }
}

void PacketProcessor::serverAgentGuildStorageDataReceived(const packet::parsing::ServerAgentGuildStorageData &packet) const {
  selfEntity_->setGuildStorageGold(packet.gold());
  helpers::initializeInventory(selfEntity_->guildStorage, packet.storageSize(), packet.storageItemMap());
  eventBroker_.publishEvent(event::EventCode::kGuildStorageInitialized);
}

void PacketProcessor::clientAgentActionCommandRequestReceived(const packet::parsing::ClientAgentActionCommandRequest &packet) const {
  CHAR_LOG_IF(INFO, absl::GetFlag(FLAGS_log_skills)) << "<Packet> ClientAgentActionCommandRequest: " << packet.actionCommand().toString();
  selfEntity_->skillEngine.pendingCommandQueue.push_back(packet.actionCommand());
  printCommandQueues();
}

void PacketProcessor::serverAgentActionCommandResponseReceived(const packet::parsing::ServerAgentActionCommandResponse &packet) const {
  CHAR_LOG_IF(INFO, absl::GetFlag(FLAGS_log_skills)) << "<Packet> ServerAgentActionCommandResponse: " << packet.actionState();

  if (packet.actionState() == packet::enums::ActionState::kQueued) {
    if (selfEntity_->skillEngine.pendingCommandQueue.empty()) {
      throw std::runtime_error("Command queued, but pending command list is empty");
    }
    CHAR_LOG_IF(INFO, absl::GetFlag(FLAGS_log_skills)) << "Command accepted: " << selfEntity_->skillEngine.pendingCommandQueue.front().toString();
    selfEntity_->skillEngine.acceptedCommandQueue.emplace_back(selfEntity_->skillEngine.pendingCommandQueue.front());
    selfEntity_->skillEngine.pendingCommandQueue.erase(selfEntity_->skillEngine.pendingCommandQueue.begin());
    const auto &command = selfEntity_->skillEngine.acceptedCommandQueue.back();
    if (command.command.commandType == packet::enums::CommandType::kExecute &&
        command.command.actionType == packet::enums::ActionType::kCast) {
      const auto &skillData = gameData_.skillData().getSkillById(command.command.refSkillId);
      CHAR_LOG_IF(INFO, absl::GetFlag(FLAGS_log_skills)) << "Queued command is to cast skill " << gameData_.getSkillName(command.command.refSkillId);
    }
    if (!selfEntity_->skillEngine.pendingCommandQueue.empty()) {
      CHAR_LOG_IF(INFO, absl::GetFlag(FLAGS_log_skills)) << "There are " << selfEntity_->skillEngine.pendingCommandQueue.size() << " more commands in the pending queue";
    }
  } else if (packet.actionState() == packet::enums::ActionState::kError) {
    // 16388 happens when the skill is on cooldown
    if (selfEntity_->skillEngine.pendingCommandQueue.empty()) {
      throw std::runtime_error("Command error, but pending command list is empty");
    }
    // Error seems to always refer to the most recent
    const auto failedCommand = selfEntity_->skillEngine.pendingCommandQueue.front();
    CHAR_LOG_IF(INFO, absl::GetFlag(FLAGS_log_skills)) << "Command error " << packet.errorCode() << ": " << wrapActionCommand(failedCommand);
    eventBroker_.publishEvent<event::CommandError>(selfEntity_->globalId, failedCommand);
    selfEntity_->skillEngine.pendingCommandQueue.erase(selfEntity_->skillEngine.pendingCommandQueue.begin());
  } else if (packet.actionState() == packet::enums::ActionState::kEnd) {
    // It seems like if a skill is completed without interruption, this end will come after the SkillEnd packet
    // If a skill is interrupted, this end will come BEFORE the SkillEnd packet
    if (selfEntity_->skillEngine.acceptedCommandQueue.empty()) {
      bool poppedACancel=false;
      if (!selfEntity_->skillEngine.pendingCommandQueue.empty()) {
        VLOG(1) << " Pending command queue is not empty though, maybe we ought to pop that?";
        if (selfEntity_->skillEngine.pendingCommandQueue.front().commandType == packet::enums::CommandType::kCancel) {
          VLOG(1) << "  Action is a cancel, popping";
          // TODO: I am not confident in the assumption that this means that we delete the first item in the pending queue
          selfEntity_->skillEngine.pendingCommandQueue.erase(selfEntity_->skillEngine.pendingCommandQueue.begin());
          poppedACancel = true;
        }
      }
      if (!poppedACancel) {
        CHAR_LOG(WARNING) << "Command ended, but we had no accepted command";
      }
    } else {
      // We arent told which command ended, we just assume it was the most recent.
      // If we have multiple, it could actually be the most recent.
      const auto firstCommandIt = selfEntity_->skillEngine.acceptedCommandQueue.begin();
      CHAR_LOG_IF(INFO, absl::GetFlag(FLAGS_log_skills)) << selfEntity_->name << " Command end: "  << firstCommandIt->command.toString();
      if (firstCommandIt->command.commandType == packet::enums::CommandType::kExecute &&
          firstCommandIt->command.actionType == packet::enums::ActionType::kCast &&
          !firstCommandIt->wasExecuted) {
        VLOG(1) << "This command (skill " << gameData_.getSkillName(firstCommandIt->command.refSkillId) << "(" << firstCommandIt->command.refSkillId << ")) was never executed!!";
        eventBroker_.publishEvent<event::CommandSkipped>(selfEntity_->globalId, firstCommandIt->command);
      }
      if (firstCommandIt->command.commandType == packet::enums::CommandType::kExecute &&
          firstCommandIt->command.actionType == packet::enums::ActionType::kCast) {
        CHAR_LOG_IF(INFO, absl::GetFlag(FLAGS_log_skills)) << "Skill " << gameData_.getSkillName(firstCommandIt->command.refSkillId) << "(" << firstCommandIt->command.refSkillId << ") command ended";
      }
      if (firstCommandIt->command.commandType == packet::enums::CommandType::kExecute &&
          firstCommandIt->command.actionType == packet::enums::ActionType::kDispel) {
        // Dispel success.
      }
      selfEntity_->skillEngine.acceptedCommandQueue.erase(firstCommandIt);
    }
  } else {
    throw std::runtime_error("Impossible action state");
  }

  printCommandQueues();
}

WrappedCommand PacketProcessor::wrapActionCommand(const packet::structures::ActionCommand &command) const {
  return WrappedCommand(command, gameData_);
}

void PacketProcessor::printCommandQueues() const {
  std::stringstream ss;
  ss << "Pending command Queue:" << (selfEntity_->skillEngine.pendingCommandQueue.empty() ? " <empty>" : "") << '\n';
  for (const auto &c : selfEntity_->skillEngine.pendingCommandQueue) {
    ss << "  " << wrapActionCommand(c) << '\n';
  }
  ss << "Accepted command Queue:" << (selfEntity_->skillEngine.acceptedCommandQueue.empty() ? " <empty>" : "") << '\n';
  for (const auto &c : selfEntity_->skillEngine.acceptedCommandQueue) {
    ss << "  [" << (c.wasExecuted ? 'X' : ' ') << "] " << wrapActionCommand(c.command) << '\n';
  }
  ss << "--------------------------------------------------------------------------------------------------";
  CHAR_LOG_IF(INFO, absl::GetFlag(FLAGS_log_skills)) << " - - - - Printing command queue - - - -\n" << ss.str();
}

// Skill Notes:
/*
ActionCastingTime is the amount of time it takes for the skill to "end" after it began
ActionActionDuration is the amount of time it takes for the skill to cast before the character is free
ActionReuseDelay is the skill's cooldown
*/

void PacketProcessor::serverAgentSkillBeginReceived(const packet::parsing::ServerAgentSkillBegin &packet) const {
  const broker::EventBroker::ClockType::time_point currentTime = broker::EventBroker::ClockType::now();
  CHAR_LOG_IF(INFO, absl::GetFlag(FLAGS_log_skills)) << "<Packet> ServerAgentSkillBeginReceived";
  if (packet.result() == 2) {
    // Error
    CHAR_LOG_IF(INFO, absl::GetFlag(FLAGS_log_skills)) << "Skill unsuccessful, err " << packet.errorCode();

    bool found=false;
    // Which skill failed? Let's look to see if there is something in the accepted command queue.
    for (const state::SkillEngine::AcceptedCommandAndWasExecuted &command : selfEntity_->skillEngine.acceptedCommandQueue) {
      if (command.command.commandType == packet::enums::CommandType::kExecute &&
          command.command.actionType == packet::enums::ActionType::kCast) {
        CHAR_LOG_IF(INFO, absl::GetFlag(FLAGS_log_skills)) << absl::StreamFormat("Found skill %s in accepted command queue", gameData_.getSkillName(command.command.refSkillId));
        if (!found) {
          CHAR_LOG_IF(INFO, absl::GetFlag(FLAGS_log_skills)) << "  Publishing skill failed event";
          eventBroker_.publishEvent<event::SkillFailed>(selfEntity_->globalId, command.command.refSkillId, packet.errorCode());
          found = true;
        }
      }
    }
    for (const packet::structures::ActionCommand &command : selfEntity_->skillEngine.pendingCommandQueue) {
      if (command.commandType == packet::enums::CommandType::kExecute &&
          command.actionType == packet::enums::ActionType::kCast) {
        CHAR_LOG_IF(INFO, absl::GetFlag(FLAGS_log_skills)) << absl::StreamFormat("Found skill %s in pending command queue", gameData_.getSkillName(command.refSkillId));
        if (!found) {
          CHAR_LOG_IF(INFO, absl::GetFlag(FLAGS_log_skills)) << "  Publishing skill failed event";
          eventBroker_.publishEvent<event::SkillFailed>(selfEntity_->globalId, command.refSkillId, packet.errorCode());
          found = true;
        }
      }
    }
    return;
  }

  // Update world state based on action
  handleSkillAction(packet.action(), packet.casterGlobalId());

  // Do some skill tracking work
  if (packet.casterGlobalId() == selfEntity_->globalId) {
    // BEGIN DEBUGGING SkillBegin/SkillEnd
    // Track this mf
    const bool expectEnd = [&]() -> bool {
      const auto &skillData = gameData_.skillData().getSkillById(packet.refSkillId());
      if (skillData.param1Type() == sro::pk2::ref::Skill::Param1Type::kBuff) {
        // It seems that buffs always have an end.
        return true;
      } else {
        // Is a chain?
        const auto rootRef = gameData_.skillData().getRootSkillRefId(packet.refSkillId());
        const bool isRoot = rootRef == packet.refSkillId();
        const bool isChain = skillData.basicChainCode != 0 || !isRoot;
        if (isChain) {
          return (skillData.actionPreparingTime + skillData.actionCastingTime > 0);
        } else {
          return (skillData.actionPreparingTime + skillData.actionCastingTime > 0);
        }
      }
      throw std::runtime_error("Unimplemented");
    }();
    auto logNoEnd = [&](const auto skillRefId) {
      // LOG(INFO) << std::string(1000, 'L') << "\nNo end came for skill " << skillRefId << " (" << gameData_.getSkillName(skillRefId) << ")";
      // std::ofstream myFile("no_end.txt", std::ios::app);
      // if (myFile) {
      //   myFile << skillRefId << ',';
      // }
    };
    // Do we already have a tracked begin for this skill? That would mean that an end never came
    for (auto it=tracked_.begin(); it!=tracked_.end();) {
      if (it->second.refSkillId == packet.refSkillId()) {
        // We already tracked this skill. This means an end never came
        if (expectEnd) {
          logNoEnd(it->second.refSkillId);
        }
        it = tracked_.erase(it);
      } else {
        ++it;
      }
    }
    auto &thing = tracked_[packet.castId()];
    thing.refSkillId = packet.refSkillId();;
    thing.casterGlobalId = packet.casterGlobalId();;
    thing.expTime = std::chrono::high_resolution_clock::now() + std::chrono::seconds(10);
    thing.expectEnd = expectEnd;
    // Run through all of these and see if any expired on time
    for (auto it=tracked_.begin(); it!=tracked_.end();) {
      auto &trackedItem = it->second;
      if (trackedItem.expTime <= std::chrono::high_resolution_clock::now()) {
        // This one expired
        if (trackedItem.expectEnd) {
          logNoEnd(trackedItem.refSkillId);
        }
        it = tracked_.erase(it);
      } else {
        ++it;
      }
    }
    // END DEBUGGING SkillBegin/SkillEnd

    // Only tracking our own skills for now
    const auto &skillData = gameData_.skillData().getSkillById(packet.refSkillId());
    const auto rootSkillRefId = gameData_.skillData().getRootSkillRefId(packet.refSkillId());
    const bool isRootSkill = rootSkillRefId == packet.refSkillId();
    const bool skillIsCommonAttack = (skillData.basicCode.find("BASE") != std::string::npos ||
                                      skillData.basicCode.find("PUNCH") != std::string::npos); // TODO: Find a more robust way to check
    auto skillName = [&]() -> std::string {
      if (skillIsCommonAttack) {
        return "Common";
      }
      return gameData_.getSkillName(packet.refSkillId());
    }();
    CHAR_LOG_IF(INFO, absl::GetFlag(FLAGS_log_skills)) << "SkillBegin \"" << skillName << "\" (" << packet.refSkillId() << ") with preparing time: " << skillData.actionPreparingTime << ", casting time: " << skillData.actionCastingTime << ", action duration: " << skillData.actionActionDuration << ", and reuse delay: " << skillData.actionReuseDelay;
    // if (!isRootSkill) {
    //   LOG(INFO) << "  Skill " << packet.refSkillId() << "'s root is " << rootSkillRefId;
    // }
    if (isRootSkill) {
      // Don't send skill begin packets for skills which are in the middle of a chain.
      // LOG(INFO) << "Publishing skill began event";
      eventBroker_.publishEvent<event::SkillBegan>(selfEntity_->globalId, packet.refSkillId());
    }
    // We expect that were is at least one accepted command in the queue
    const bool isFinalPieceOfChain = skillData.basicChainCode == 0;
    if (!selfEntity_->skillEngine.acceptedCommandQueue.empty()) {
      // Try to find the index of this skill in the accepted command queue.
      std::optional<size_t> indexOfOurSkill;
      auto &acceptedCommandQueue = selfEntity_->skillEngine.acceptedCommandQueue;
      // Walk in reverse order of the queue looking for a skill which matched out. The last item is the newest. Later, when we remove items after our command, it will at worst be an underestimate of the number of things to remove.
      for (int i=acceptedCommandQueue.size()-1; i>=0; --i) {
        const auto &acceptedCommand = acceptedCommandQueue.at(i);
        if (acceptedCommand.command.actionType == packet::enums::ActionType::kCast && acceptedCommand.command.refSkillId == rootSkillRefId) {
          indexOfOurSkill = i;
          break;
        } else if (acceptedCommand.command.actionType == packet::enums::ActionType::kAttack) {
          // Must be a common attack.
          if (skillIsCommonAttack) {
            indexOfOurSkill = i;
            break;
          }
        }
      }
      // TODO: Common attacks will not be found, does that matter? That means that no skill cooldown will be created for one
      if (!indexOfOurSkill) {
        // TODO: Shouldn't happen now
        CHAR_LOG(WARNING) << absl::StreamFormat("Couldn't find our skill in the accepted command queue. packet.refSkillId(): %d, rootSkillRefId: %d", packet.refSkillId(), rootSkillRefId);
        // This happens for common attacks
        if (skillIsCommonAttack && !(acceptedCommandQueue.front().command.commandType == packet::enums::CommandType::kExecute &&
                                     acceptedCommandQueue.front().command.actionType == packet::enums::ActionType::kAttack)) {
          // First command is not a common attack
          // LOG(INFO) << "First command in the queue isn't a common attack!";
        }
      } else {
        // We cast this skill
        if (*indexOfOurSkill != 0) {
          // Remove all commands before this one in the queue, those have probably been discarded by the server
          VLOG(1) << "Our skill is not the first in the accepted command queue.";
          // TODO: Should previous items be removed?

          //  We should in this case:
          //    1. Cast "Flying Dragon - Flash"
          //    2. Get a "queued" response
          //    3. Cast "Snow Shield - Intensify"
          //    4. Get a "queued" response
          //    5. Get a "SkillBegin" packet
          //    6. The "SkillBegin" packet is for "Snow Shield - Intensify"

          //  We should not in this case:
          //    1. Cast "Snow Shield - Intensify"
          //    2. Get a "queued" response
          //    3. Get a "SkillBegin" packet
          //    4. Cast "Thunder Phoenix Force"
          //    5. Get a "queued" response
          //    6. Get a "SkillBegin" packet (for Thunder Phoenix Force)

          if (skillData.basicActivity == 1) {
            // Can be executed before other skills in the queue. Do not delete predecessors in the queue.
          } else {
            for (int i=0; i<*indexOfOurSkill; ++i) {
              // TODO: Should we publish an event that command has been skipped?
              VLOG(1) << "Command #" << i << " (" << wrapActionCommand(acceptedCommandQueue.at(i).command) << ") skipped";
            }
            acceptedCommandQueue.erase(acceptedCommandQueue.begin(), acceptedCommandQueue.begin() + *indexOfOurSkill);
            indexOfOurSkill = 0;
          }
        }
        // A skill always has a begin, but might not have an end.
        //  Marking this skill as executed here is sufficient
        acceptedCommandQueue.at(*indexOfOurSkill).wasExecuted = true;
        if (isRootSkill && !skillIsCommonAttack) {
          // Set a timer for when the skill cooldown ends. We only do this for the root piece of the skill. If this is a chain and later piece has a cooldown too, it is probably just the same cooldown that we already set a timer for when we cast the root.
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
          // TODO: Move the sending of this event into the entity!
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
          selfEntity_->skillCooldownBegin(packet.refSkillId(), currentTime + std::chrono::milliseconds(skillData.actionReuseDelay));
        }
        if (skillData.basicActivity == 1) {
          // No "End" will come for Basic_Activity == 1, delete the item from the accepted command queue
          acceptedCommandQueue.erase(acceptedCommandQueue.begin() + *indexOfOurSkill);
        }
        // printCommandQueues(); // COMMAND_QUEUE_DEBUG
      }
    } else {
      // This happens when we spawn in with a speed scroll
      // LOG(INFO) << "WARNING: accepted command queue empty";
    }
    bool expectSkillEnd{false};
    if (skillData.param1Type() == sro::pk2::ref::Skill::Param1Type::kBuff) {
      // SkillEnd will always come for buffs
      expectSkillEnd = true;
    } else if (skillData.actionPreparingTime + skillData.actionCastingTime > 0) {
      expectSkillEnd = true;
    }
    if (expectSkillEnd) {
      // We cast this skill, save the cast ID so that we can reference it later on SkillEnd
      selfEntity_->skillEngine.skillCastIdMap.emplace(std::piecewise_construct, std::forward_as_tuple(packet.castId()), std::forward_as_tuple(packet.casterGlobalId(), packet.refSkillId()));
      if (selfEntity_->skillEngine.skillCastIdMap.size() > 1) {
        CHAR_LOG_IF(INFO, absl::GetFlag(FLAGS_log_skills)) << absl::StreamFormat("Skill casts tracked: [ %s ]", absl::StrJoin(selfEntity_->skillEngine.skillCastIdMap, ", ", [](std::string *out, const auto data){
          out->append(std::to_string(data.first));
        }));
      }
    } else {
      // No skill end will come
      bool killedTarget = skillActionKilledTarget(packet.targetGlobalId(), packet.action());
      if (isFinalPieceOfChain || killedTarget) {
        eventBroker_.publishEvent<event::SkillEnded>(selfEntity_->globalId, packet.refSkillId());
      }
    }
  } else {
    // Caster is not us
    std::shared_ptr<entity::Entity> entity = worldState_.getEntity(packet.casterGlobalId());
    if (auto *monster = dynamic_cast<entity::Monster*>(entity.get()); monster != nullptr) {
      // Caster is a monster, track who the monster is targeting
      monster->targetGlobalId = packet.targetGlobalId();
    }
  }

  if (!gameData_.skillData().haveSkillWithId(packet.refSkillId())) {
    throw std::runtime_error("Cast a skill which we dont have data for");
  }
  const auto &skill = gameData_.skillData().getSkillById(packet.refSkillId());
  switch (skill.basicActivity) {
    case 0:
      // Seems to be passives
      // LOG(INFO) << "Cast a skill with basic activity == 0";
      break;
    case 1:
      // Dont stop while running. Can be cast while something else is being case
      break;
    case 2:
      {
        // Will stop you if you're running
        std::shared_ptr<entity::MobileEntity> casterAsMobileEntity = worldState_.getEntity<entity::MobileEntity>(packet.casterGlobalId());
        if (casterAsMobileEntity->moving()) {
          casterAsMobileEntity->setStationaryAtPosition(casterAsMobileEntity->position());
        }
        break;
      }
    default:
      throw std::runtime_error("Cast a skill with unknown basic activity == "+std::to_string(skill.basicActivity));
      break;
  }
}

void PacketProcessor::serverAgentSkillEndReceived(const packet::parsing::ServerAgentSkillEnd &packet) const {
  CHAR_LOG_IF(INFO, absl::GetFlag(FLAGS_log_skills)) << "<Packet> ServerAgentSkillEnd";
  // BEGIN DEBUGGING SkillBegin/SkillEnd
  {
    auto it = tracked_.find(packet.castId());
    if (it != tracked_.end()) {
      const auto &trackedSkill = it->second;
      if (trackedSkill.casterGlobalId == selfEntity_->globalId) {
        // Only want to do this checking for our own skills
        if (!trackedSkill.expectEnd) {
          const auto skillRefId = trackedSkill.refSkillId;
          LOG(INFO) << std::string(1000, 'L') << "\nUnexpected end came for skill " << skillRefId << " (" << gameData_.getSkillName(skillRefId) << ")";
          std::ofstream myFile("unexpected_end.txt", std::ios::app);
          if (myFile) {
            myFile << skillRefId << ',';
          }
        }
      } else {
        LOG(INFO) << "We tracked a skill which we did not cast. Weird.";
      }
      // Remove from map.
      tracked_.erase(it);
    }
  }
  // END DEBUGGING SkillBegin/SkillEnd
  if (packet.result() == 2) {
    // Not successful?
    VLOG(1) << absl::StreamFormat("Result == %d, error: %d. Cast ID %d", packet.result(), packet.errorCode(), packet.castId());
    if (packet.errorCode() != 0) {
      CHAR_LOG(WARNING) << "Newly seen error code!";
    }
    auto skillCastIt = selfEntity_->skillEngine.skillCastIdMap.find(packet.castId());
    if (skillCastIt != selfEntity_->skillEngine.skillCastIdMap.end()) {
      VLOG(2) << "Publishing a successful end";
      eventBroker_.publishEvent<event::SkillEnded>(selfEntity_->globalId, skillCastIt->second.skillRefId);
      VLOG(2) << "Deleting tracked cast";
      selfEntity_->skillEngine.skillCastIdMap.erase(skillCastIt);
    } else {
      CHAR_LOG(WARNING) << "Skill ended with error, but we didn't track the cast";
    }
    return;
  }

  std::optional<sro::scalar_types::EntityGlobalId> casterGlobalId;
  auto skillCastIt = selfEntity_->skillEngine.skillCastIdMap.find(packet.castId());
  if (skillCastIt != selfEntity_->skillEngine.skillCastIdMap.end()) {
    // We cast this skill.
    casterGlobalId = selfEntity_->globalId;
  }
  handleSkillAction(packet.action(), casterGlobalId);

  // Have we tracked this skill?
  if (skillCastIt != selfEntity_->skillEngine.skillCastIdMap.end()) {
    auto &skillInfo = skillCastIt->second;
    const auto thisSkillId = skillInfo.skillRefId;
    if (skillInfo.casterGlobalId == selfEntity_->globalId) {
      // We cast this skill
      const auto &skillData = gameData_.skillData().getSkillById(thisSkillId);
      VLOG(4) << "  Is our \"" << gameData_.getSkillName(thisSkillId) << "\" end";
      bool doneWithSkill{true};
      if (skillData.basicChainCode != 0) {
        // There are more pieces to this skill
        skillInfo.skillRefId = skillData.basicChainCode;
        // TODO: afaict, we should always not be done with the skill when there are more parts of the chain coming
        doneWithSkill = false;
        // const auto &nextSkillPiece = gameData_.skillData().getSkillById(skillInfo.skillRefId);
        // if (nextSkillPiece.actionCastingTime != 0) {
        //   LOG(INFO) << "  Another piece of skill is coming, basic chain code=" << skillData.basicChainCode << ". This skill has non-zero cast time: " << nextSkillPiece.actionCastingTime;
        // } else {
        //   LOG(INFO) << "  Skill has another piece, but has 0 casting time.";
        //   // TODO: Check if there is ANOTHER piece coming...
        // }
      }
      const bool killedTarget = skillActionKilledTarget(packet.targetGlobalId(), packet.action());
      doneWithSkill |= killedTarget;
      if (doneWithSkill) {
        // Publish skill end event.
        eventBroker_.publishEvent<event::SkillEnded>(selfEntity_->globalId, thisSkillId);
        // Remove it from the map
        selfEntity_->skillEngine.skillCastIdMap.erase(skillCastIt);
      }
    } else {
      // LOG(INFO) << "  Is NOT our skill end";
    }
  } else {
    VLOG(4) << "  Untracked cast";
  }
}

void PacketProcessor::handleSkillAction(const packet::structures::SkillAction &action, std::optional<sro::scalar_types::EntityGlobalId> casterGlobalId) const {
  // LOG(INFO) << "    -- Handle Skill Action";
  if (casterGlobalId &&
     (flags::isSet(action.actionFlag, packet::enums::ActionFlag::kTeleport) ||
      flags::isSet(action.actionFlag, packet::enums::ActionFlag::kSprint))) {
    // Entity teleported or sprinted to a new position. Sprints are actually teleports on the server side, even though the skill actually has a duration. The duration is only used for animation
    std::shared_ptr<entity::MobileEntity> casterAsMobileEntity = worldState_.getEntity<entity::MobileEntity>(*casterGlobalId);
    casterAsMobileEntity->setStationaryAtPosition(action.position);
  }

  auto nameOfEntity = [&](auto gid) -> std::string {
    std::shared_ptr<entity::Entity> entity = worldState_.getEntity(gid);
    if (auto *playerCharacter = dynamic_cast<entity::PlayerCharacter*>(entity.get())) {
      return playerCharacter->name;
    } else {
      return std::to_string(gid);
    }
  };

  for (const auto &hitObject : action.hitObjects) {
    for (const auto &hitResult : hitObject.hits) {
      if (casterGlobalId && selfEntity_ && *casterGlobalId == selfEntity_->globalId) {
        // We cast this.
      }
      if (hitResult.damage > 0 && casterGlobalId) {
        eventBroker_.publishEvent<event::DealtDamage>(*casterGlobalId, hitObject.targetGlobalId, hitResult.damage);
      }
      std::shared_ptr<entity::Entity> targetEntity = worldState_.getEntity(hitObject.targetGlobalId);
      if (auto *character = dynamic_cast<entity::Character*>(targetEntity.get())) {
        if (flags::isSet(hitResult.hitResultFlag, packet::enums::HitResult::kKill)) {
          // Effectively killed it, but I don't know if it makes sense to change the life state right now
          character->setCurrentHp(0);
        } else {
          if (flags::isSet(hitResult.hitResultFlag, packet::enums::HitResult::kKnockdown)) {
            VLOG(1) << "      Entity has been knocked down";
            // TODO: Update entity state, publish knocked down event(?), publish delayed stood up event
          }
          if (flags::isSet(hitResult.damageFlag, packet::enums::DamageFlag::kEffect) && hitResult.effect != 0) {
            if (selfEntity_ && hitObject.targetGlobalId == selfEntity_->globalId) {
              // Applied an effect to us
              VLOG(1) << "      We have been hit with effect: " << static_cast<packet::enums::AbnormalStateFlag>(hitResult.effect);
            } else {
              VLOG(1) << "      " << nameOfEntity(hitObject.targetGlobalId) << " has been hit with effect: " << static_cast<packet::enums::AbnormalStateFlag>(hitResult.effect);
            }
          }
          if (character->currentHpIsKnown()) {
            // Can only update the character's hp if we know what it currently is
            if (hitResult.damage >= character->currentHp()) {
              character->setCurrentHp(0);
            } else {
              character->setCurrentHp(std::max<int64_t>(0, int64_t(character->currentHp()) - hitResult.damage));
            }
          }
        }
      }
      if (flags::isSet(hitResult.hitResultFlag, packet::enums::HitResult::kKnockback) ||
          flags::isSet(hitResult.hitResultFlag, packet::enums::HitResult::kKnockdown)) {
        if (auto *targetAsMobileEntity = dynamic_cast<entity::MobileEntity*>(targetEntity.get())) {
          targetAsMobileEntity->setStationaryAtPosition(hitResult.position);
          if (selfEntity_ && hitObject.targetGlobalId == selfEntity_->globalId) {
            // We are the target
            bool knockedBackOrKnockedDown{false};
            if (flags::isSet(hitResult.hitResultFlag, packet::enums::HitResult::kKnockback)) {
              VLOG(1) << "      We were knocked back " << static_cast<int>(hitResult.hitResultFlag) << ", sending stun delayed event " << sro::game_constants::kKnockbackStunDuration.count() << "ms";
              selfEntity_->stunnedFromKnockback = true;
              knockedBackOrKnockedDown = true;
              // Publish knocked back event
              eventBroker_.publishEvent<event::KnockedBack>(selfEntity_->globalId);
              // Publish delayed knocked back stun completed event
              eventBroker_.publishDelayedEvent<event::KnockbackStunEnded>(sro::game_constants::kKnockbackStunDuration, selfEntity_->globalId);
            } else if (flags::isSet(hitResult.hitResultFlag, packet::enums::HitResult::kKnockdown)) {
              VLOG(1) << "      We were knocked down " << static_cast<int>(hitResult.hitResultFlag) << ", sending stun delayed event " << sro::game_constants::kKnockdownStunDuration.count() << "ms";
              selfEntity_->stunnedFromKnockdown = true;
              knockedBackOrKnockedDown = true;
              // Publish knocked down event
              eventBroker_.publishEvent<event::KnockedDown>(selfEntity_->globalId);
              // Publish delayed knocked down stun completed event
              eventBroker_.publishDelayedEvent<event::KnockdownStunEnded>(sro::game_constants::kKnockdownStunDuration, selfEntity_->globalId);
            }
            if (knockedBackOrKnockedDown) {
              // Whatever commands we had queued should probably be cleared
              // Whatever skills were casting should probably be cancelled
              handleKnockedBackOrKnockedDown();
            }
          }
        } else {
          throw std::runtime_error("Knockback/knockdown on non-mobile entity");
        }
      }
    }
  }
}

void PacketProcessor::handleKnockedBackOrKnockedDown() const {
  // When knocked back or knocked down, all accepted commands are not going to execute
  selfEntity_->skillEngine.acceptedCommandQueue.clear();
  // It doesn't make sense to remove all pending commands as those have not even been acknowledge by the server yet
  //  The server will likely respond with an error response for them
  if (!selfEntity_->skillEngine.skillCastIdMap.empty()) {
    VLOG(1) << absl::StreamFormat("KB/KD with active casts: %s. Clearing", absl::StrJoin(selfEntity_->skillEngine.skillCastIdMap, ", ", [](std::string *out, auto data){
      out->append(std::to_string(data.first));
    }));
    // TODO: Verify if it makes sense to clear this
    //  It certainly seems like all started casts will be interrupted
    selfEntity_->skillEngine.skillCastIdMap.clear();
  }
}

void PacketProcessor::serverAgentBuffAddReceived(const packet::parsing::ServerAgentBuffAdd &packet) const {
  entity::Character::BuffData::ClockType::time_point currentTime = entity::Character::BuffData::ClockType::now();
  const auto skillName = gameData_.getSkillName(packet.skillRefId());
  if (packet.activeBuffToken() == 0) {
    // No buff remove will be received when this expires
    //  Seems to be only for debuffs
    //  Weirdly, it's also sent for Sprint Assault and Integrity.
    VLOG(1) << "Skipping buff \"" << skillName << "\" for " << worldState_.getEntity(packet.globalId())->toString() << " with tokenId: " << packet.activeBuffToken();
    return;
  }
  VLOG(1) << "Buff \"" << skillName << "(" << packet.skillRefId() << ")\" added to " << worldState_.getEntity(packet.globalId())->toString() << " with tokenId: " << packet.activeBuffToken();
  worldState_.addBuff(packet.globalId(), packet.skillRefId(), packet.activeBuffToken(), currentTime);
}

void PacketProcessor::serverAgentBuffLinkReceived(const packet::parsing::ServerAgentBuffLink &packet) const {
  VLOG(1) << "Buff link received " << gameData_.getSkillName(packet.skillRefId()) << "(" << packet.skillRefId() << ")," << packet.activeBuffToken() << ',' << packet.targetGlobalId() << ',' << packet.targetName();
  // TODO: Where should I track the buff link? It seems to be a duplicate of what was sent in the "BuffAdd" packet.
}

void PacketProcessor::serverAgentBuffRemoveReceived(const packet::parsing::ServerAgentBuffRemove &packet) const {
  VLOG(1) << absl::StreamFormat("Buff remove received. Buffs to remove: [ %s ]", absl::StrJoin(packet.tokens(), ", ", [](std::string *out, auto tokenId){
    out->append(std::to_string(tokenId));
  }));
  worldState_.removeBuffs(packet.tokens());
}

void PacketProcessor::serverAgentChatUpdateReceived(const packet::parsing::ServerAgentChatUpdate &packet) const {
  if (packet.chatType() == packet::enums::ChatType::kAll ||
      packet.chatType() == packet::enums::ChatType::kAllGm ||
      packet.chatType() == packet::enums::ChatType::kNpc) {
    eventBroker_.publishEvent<event::ChatReceived>(packet.chatType(), packet.senderGlobalId(), packet.message());
  } else {
    eventBroker_.publishEvent<event::ChatReceived>(packet.chatType(), packet.senderName(), packet.message());
  }
}

void PacketProcessor::serverAgentGameResetReceived(const packet::parsing::ServerAgentGameReset &packet) {
  worldState_.entityDespawned(selfEntity_->globalId, eventBroker_);
  selfEntity_.reset();
}

void PacketProcessor::serverAgentResurrectOptionReceived(const packet::parsing::ServerAgentResurrectOption &packet) const {
  eventBroker_.publishEvent<event::ResurrectOption>(packet.option());
}

void PacketProcessor::serverAgentOperatorResponseReceived(const packet::parsing::ServerAgentOperatorResponse &packet) const {
  if (packet.result() == 1) {
    eventBroker_.publishEvent<event::OperatorRequestSuccess>(selfEntity_->globalId, packet.operatorCommand());
  } else if (packet.result() == 2) {
    eventBroker_.publishEvent<event::OperatorRequestError>(selfEntity_->globalId, packet.operatorCommand());
  } else {
    throw std::runtime_error("Unknown operator response result");
  }
}

void PacketProcessor::serverAgentFreePvpUpdateResponseReceived(const packet::parsing::ServerAgentFreePvpUpdateResponse &packet) const {
  if (packet.result() == 1) {
    if (packet.globalId() == selfEntity_->globalId) {
      selfEntity_->freePvpMode = packet.mode();
      eventBroker_.publishEvent<event::FreePvpUpdateSuccess>(packet.globalId());
    }
  }
}

void PacketProcessor::serverAgentInventoryEquipCountdownStartReceived(const packet::parsing::ServerAgentInventoryEquipCountdownStart &packet) const {
  if (packet.globalId() == selfEntity_->globalId) {
    eventBroker_.publishEvent<event::EquipCountdownStart>(packet.globalId());
  }
}

std::string PacketProcessor::characterNameForLog() const {
  if (selfEntity_ == nullptr) {
    return absl::StrFormat("[NOT_LOGGED_IN]");
  } else {
    return absl::StrFormat("[%s]", selfEntity_->name);
  }
}