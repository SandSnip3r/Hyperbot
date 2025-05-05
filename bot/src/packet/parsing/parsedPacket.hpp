#include "entity/entity.hpp"
#include "packet/opcode.hpp"
#include "packet/enums/packetEnums.hpp"
#include "packet/structures/packetInnerStructures.hpp"
#include "pk2/characterData.hpp"
#include "pk2/gameData.hpp"
#include "pk2/itemData.hpp"
#include "pk2/skillData.hpp"
#include "pk2/teleportData.hpp"
#include "shared/silkroad_security.h"
#include "storage/item.hpp"

#include <silkroad_lib/position.hpp>

#include <array>
#include <map>
#include <memory>
#include <string>

#ifndef PACKET_PARSING_HPP
#define PACKET_PARSING_HPP

namespace packet::parsing {

//=========================================================================================================================================================

class ParsedPacket {
protected:
  const Opcode opcode_;
public:
  ParsedPacket(const PacketContainer &packet);
  Opcode opcode() const;
  virtual ~ParsedPacket() = 0;
};

//=========================================================================================================================================================

class ParsedClientItemMove : public ParsedPacket {
public:
  ParsedClientItemMove(const PacketContainer &packet);
  structures::ItemMovement movement() const;
private:
  structures::ItemMovement movement_;
};

//=========================================================================================================================================================

} // namespace packet::parsing

#endif // PACKET_PARSING_HPP