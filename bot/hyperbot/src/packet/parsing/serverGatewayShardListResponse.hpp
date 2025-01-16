#ifndef PACKET_PARSING_SERVER_GATEWAY_SHARD_LIST_RESPONSE_HPP_
#define PACKET_PARSING_SERVER_GATEWAY_SHARD_LIST_RESPONSE_HPP_

#include "packet/parsing/parsedPacket.hpp"
#include "packet/structures/packetInnerStructures.hpp"

#include <vector>

namespace packet::parsing {

class ServerGatewayShardListResponse : public ParsedPacket {
public:
  ServerGatewayShardListResponse(const PacketContainer &packet);
  const std::vector<structures::Shard>& shards() const { return shards_; }
private:
  std::vector<structures::Shard> shards_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_GATEWAY_SHARD_LIST_RESPONSE_HPP_