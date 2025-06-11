#ifndef PACKET_PARSING_PACKET_PARSER_H_
#define PACKET_PARSING_PACKET_PARSER_H_

#include "parsedPacket.hpp"
#include "shared/silkroad_security.h"
#include "state/entityTracker.hpp"

#include <silkroad_lib/pk2/gameData.hpp>

#include <memory>

namespace packet::parsing {

class PacketParser {
public:
  PacketParser(const state::EntityTracker &entityTracker, const sro::pk2::GameData &gameData);
  std::unique_ptr<ParsedPacket> parsePacket(const PacketContainer &packet) const;
private:
  const state::EntityTracker &entityTracker_;
  const sro::pk2::GameData &gameData_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_PACKET_PARSER_H_
