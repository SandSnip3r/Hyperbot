#include "clientAgentCharacterMoveRequest.hpp"
#include "../enums/packetEnums.hpp"

namespace packet::building {

// https://www.elitepvpers.com/forum/sro-coding-corner/1992345-coordinate-converter-open-source.html#post17651107
// 1   bool    HasDestination
// if(HasDestination)
// {
//     2   ushort  Destination.RegionID
//     if(Destination.RegionID < short.MaxValue)
//     {
//         //World
//         2   ushort  Destination.XOffset
//         2   ushort  Destination.YOffset
//         2   ushort  Destination.ZOffset
//     }
//     else
//     {
//         //Dungeon
//         4   uint  Destination.XOffset
//         4   uint  Destination.YOffset
//         4   uint  Destination.ZOffset
//     }
// }
// else
// {
//     1   byte    AngleAction
//     2   ushort  Angle
// }

// public enum AngleAction : byte
// {
//     Obsolete = 0, //GO_BACKWARDS or SPIN?
//     GoForward = 1
// }

enum class HasDestination : uint8_t {
  kNo = 0,
  kYes = 1
};

PacketContainer ClientAgentCharacterMoveRequest::packet(uint16_t angle) {
  StreamUtility stream;
  stream.Write<uint8_t>(static_cast<uint8_t>(HasDestination::kNo));
  stream.Write<uint8_t>(static_cast<uint8_t>(packet::enums::AngleAction::kGoForward));
  stream.Write<uint16_t>(angle);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

PacketContainer ClientAgentCharacterMoveRequest::packet(uint16_t regionId, uint32_t xOffset, uint32_t yOffset, uint32_t zOffset) {
  StreamUtility stream;
  stream.Write<uint8_t>(static_cast<uint8_t>(HasDestination::kYes));
  stream.Write<uint16_t>(regionId);
  if (regionId & 0x8000) {
    // Dungeon
    stream.Write<uint32_t>(xOffset);
    stream.Write<uint32_t>(yOffset);
    stream.Write<uint32_t>(zOffset);
  } else {
    // World
    stream.Write<uint16_t>(xOffset);
    stream.Write<uint16_t>(yOffset);
    stream.Write<uint16_t>(zOffset);
  }
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

} // namespace packet::building