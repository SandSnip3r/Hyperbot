#include "clientAgentCharacterMoveRequest.hpp"

#include <utility>

namespace packet::parsing {

ClientAgentCharacterMoveRequest::ClientAgentCharacterMoveRequest(const PacketContainer &packet) :
      ParsedPacket(packet) {
  // 1   bool    HasDestination
  // if(HasDestination)
  // {
  //     2   ushort  Destination.RegionID
  //     if(Destination.RegionID < short.MaxValue)
  //     {
  //         //World
  //         2   short  Destination.XOffset
  //         2   short  Destination.YOffset
  //         2   short  Destination.ZOffset
  //     }
  //     else
  //     {
  //         //Dungeon
  //         4   int  Destination.XOffset
  //         4   int  Destination.YOffset
  //         4   int  Destination.ZOffset
  //     }
  // }
  // else
  // {
  //     1   byte    AngleAction
  //     2   ushort  Angle
  // }
  StreamUtility stream = packet.data;
  bool hasDestination = stream.Read<uint8_t>();
  if (hasDestination) {
    packet::structures::Position tmpDestinationPosition;
    tmpDestinationPosition.regionId = stream.Read<uint16_t>();
    if (tmpDestinationPosition.isDungeon()) {
      // Dungeon
      tmpDestinationPosition.xOffset = stream.Read<int32_t>();
      tmpDestinationPosition.yOffset = stream.Read<int32_t>();
      tmpDestinationPosition.zOffset = stream.Read<int32_t>();
    } else {
      // World
      tmpDestinationPosition.xOffset = stream.Read<int16_t>();
      tmpDestinationPosition.yOffset = stream.Read<int16_t>();
      tmpDestinationPosition.zOffset = stream.Read<int16_t>();
    }
    destinationPosition_.emplace(std::move(tmpDestinationPosition));
  } else {
    angleAction_ = static_cast<packet::enums::AngleAction>(stream.Read<uint8_t>());
    angle_ = stream.Read<uint16_t>();
  }
}

bool ClientAgentCharacterMoveRequest::hasDestination() const {
  return destinationPosition_.has_value();
}

const packet::structures::Position& ClientAgentCharacterMoveRequest::destinationPosition() const {
  if (!destinationPosition_) {
    throw std::runtime_error("ClientAgentCharacterMoveRequest: destinationPosition does not exist");
  }
  return destinationPosition_.value();
}

packet::enums::AngleAction ClientAgentCharacterMoveRequest::angleAction() const {
  return angleAction_;
}

uint16_t ClientAgentCharacterMoveRequest::angle() const {
  return angle_;
}

} // namespace packet::parsing