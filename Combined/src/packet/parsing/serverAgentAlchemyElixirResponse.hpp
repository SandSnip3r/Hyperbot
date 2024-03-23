#ifndef PACKET_PARSING_SERVER_AGENT_ALCHEMY_ELIXIR_RESPONSE_HPP_
#define PACKET_PARSING_SERVER_AGENT_ALCHEMY_ELIXIR_RESPONSE_HPP_

#include "packet/parsing/parsedPacket.hpp"
#include "pk2/itemData.hpp"

namespace packet::parsing {

class ServerAgentAlchemyElixirResponse : public ParsedPacket {
public:
  ServerAgentAlchemyElixirResponse(const PacketContainer &packet, const pk2::ItemData &itemData);

  uint8_t result() const;
  packet::enums::AlchemyAction alchemyAction() const;
  bool success() const;
  sro::scalar_types::StorageIndexType slot() const;
  bool itemWasDestroyed() const;
  std::shared_ptr<storage::Item> item() const;
  uint16_t errorCode() const;
private:
  uint8_t result_;
  packet::enums::AlchemyAction alchemyAction_;
  uint8_t success_;
  sro::scalar_types::StorageIndexType slot_;
  uint8_t itemWasDestroyed_;
  std::shared_ptr<storage::Item> item_;
  uint16_t errorCode_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_ALCHEMY_ELIXIR_RESPONSE_HPP_