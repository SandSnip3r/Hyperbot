#ifndef PACKET_PARSING_PACKET_PARSER_H_
#define PACKET_PARSING_PACKET_PARSER_H_

#include "gameData.hpp"
#include "parsedPacket.hpp"

#include "shared/silkroad_security.h"

#include <memory>

namespace packet::parsing {

class PacketParser {
public:
  PacketParser(const pk2::media::GameData &gameData);
  std::unique_ptr<ParsedPacket> parsePacket(const PacketContainer &packet) const;
private:
  const pk2::media::GameData &gameData_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_PACKET_PARSER_H_