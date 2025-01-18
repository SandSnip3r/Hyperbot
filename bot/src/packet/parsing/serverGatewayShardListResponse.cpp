#include "serverGatewayShardListResponse.hpp"

#include <absl/log/log.h>
#include <absl/strings/str_format.h>

namespace packet::parsing {

ServerGatewayShardListResponse::ServerGatewayShardListResponse(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  while (true) {
    uint8_t haveAnotherFarmEntry = stream.Read<uint8_t>();
    if (haveAnotherFarmEntry == 0) {
      break;
    }

    uint8_t farmId = stream.Read<uint8_t>();
    std::string farmName;
    stream.Read(farmName);
    VLOG(1) << absl::StreamFormat("Farm \"%s\"", farmName);
  }

  while (true) {
    uint8_t haveAnotherShardEntry = stream.Read<uint8_t>();
    if (!haveAnotherShardEntry) {
      break;
    }
    structures::Shard &shard = shards_.emplace_back();
    stream.Read(shard.shardId);
    stream.Read(shard.shardName);
    stream.Read(shard.onlineCount);
    stream.Read(shard.capacity);
    stream.Read(shard.isOperating);
    stream.Read(shard.farmId);
    VLOG(1) << absl::StreamFormat("Shard \"%s\"", shard.shardName);
  }
}

} // namespace packet::parsing