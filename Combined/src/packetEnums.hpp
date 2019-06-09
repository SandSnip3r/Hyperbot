#ifndef PACKET_ENUMS_HPP
#define PACKET_ENUMS_HPP

namespace PacketEnums {

enum class AngleAction { kObsolete=0, kGoForward=1 };
enum class ChatType {
  kAll = 1,
  kPm = 2,
  kAllGm = 3,
  kParty = 4,
  kGuild = 5,
  kGlobal = 6,
  kNotice = 7,
  kStall = 9,
  kUnion = 11,
  kNpc = 13,
  kAcademy = 16,
};

} // namespace PacketEnums

#endif // PACKET_ENUMS_HPP