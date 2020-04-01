#ifndef PACKET_INNER_STRUCTURES_HPP
#define PACKET_INNER_STRUCTURES_HPP

#include <string>
#include <vector>

namespace packet::structures {

namespace CharacterSelection {

struct Item {
  uint32_t refId;
  uint8_t plus;
};

struct Avatar {
  uint32_t refId;
  uint8_t plus;
};

struct Character {
public:
  uint32_t refObjID;
  // uint16_t  name.Length;
  std::string name;
  uint8_t scale;
  uint8_t curLevel;
  uint64_t expOffset;
  uint16_t strength;
  uint16_t intelligence;
  uint16_t statPoint;
  uint32_t curHP;
  uint32_t curMP;
  bool isDeleting;
    uint32_t charDeleteTime;
  uint8_t guildMemberClass;
  bool isGuildRenameRequired;
    // uint16_t currentGuildName.Length
    std::string currentGuildName;
  uint8_t academyMemberClass;
  // uint8_t itemCount;
  std::vector<Item> items;
  // uint8_t avatarItemCount;
  std::vector<Avatar> avatars;
};

} // namespace CharacterSelection

namespace vitals {

struct AbnormalState {
  uint32_t totalTime;
  uint16_t timeElapsed;
  uint16_t effectOrLevel;
};

} // namespace vitals

} // namespace packet::structures

#endif // PACKET_INNER_STRUCTURES_HPP