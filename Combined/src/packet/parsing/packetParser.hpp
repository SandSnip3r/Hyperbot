#ifndef PACKET_PARSING_PACKET_PARSER_H_
#define PACKET_PARSING_PACKET_PARSER_H_

#include "parsedPacket.hpp"
#include "../../pk2/gameData.hpp"
#include "../../shared/silkroad_security.h"

#include <memory>

namespace packet::parsing {

class PacketParser {
public:
  PacketParser(const pk2::GameData &gameData);
  std::unique_ptr<ParsedPacket> parsePacket(const PacketContainer &packet) const;
private:
  const pk2::GameData &gameData_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_PACKET_PARSER_H_