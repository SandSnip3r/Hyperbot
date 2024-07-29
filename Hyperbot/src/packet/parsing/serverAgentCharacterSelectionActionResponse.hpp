#ifndef PACKET_PARSING_SERVER_AGENT_CHARACTER_SELECTION_ACTION_RESPONSE_HPP_
#define PACKET_PARSING_SERVER_AGENT_CHARACTER_SELECTION_ACTION_RESPONSE_HPP_

#include "packet/enums/packetEnums.hpp"
#include "packet/parsing/parsedPacket.hpp"
#include "packet/structures/packetInnerStructures.hpp"

#include <vector>

namespace packet::parsing {

class ServerAgentCharacterSelectionActionResponse : public ParsedPacket {
public:
  ServerAgentCharacterSelectionActionResponse(const PacketContainer &packet);
  enums::CharacterSelectionAction action() const { return action_; }
  uint8_t result() const { return result_; }
  const std::vector<structures::character_selection::Character>& characters() const { return characters_; }
  uint16_t errorCode() const { return errorCode_; }
private:
  enums::CharacterSelectionAction action_;
  uint8_t result_;
  // Success case
  std::vector<structures::character_selection::Character> characters_;
  // Error case
  uint16_t errorCode_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_CHARACTER_SELECTION_ACTION_RESPONSE_HPP_