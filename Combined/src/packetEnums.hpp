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
enum class LoginResult {
  kSuccess = 1,
  kFailed = 2,
  kOther = 3
};
enum class LoginBlockType {
  kPunishment = 1,
  kAccountInspection = 2,
  kNoAccountInfo = 3,
  kFreeServiceOver = 4
};
enum class CharacterSelectionAction {
  kCreate = 1,
  kList = 2,
  kDelete = 3,
  kCheckName = 4,
  kRestore = 5
};

} // namespace PacketEnums

#endif // PACKET_ENUMS_HPP