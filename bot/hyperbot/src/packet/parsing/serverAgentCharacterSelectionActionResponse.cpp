#include "serverAgentCharacterSelectionActionResponse.hpp"

namespace packet::parsing {

ServerAgentCharacterSelectionActionResponse::ServerAgentCharacterSelectionActionResponse(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  stream.Read(action_);
  stream.Read(result_);
  if (result_ == 1 && action_ == enums::CharacterSelectionAction::kList) {
    const int characterCount = stream.Read<uint8_t>();
    for (int i=0; i<characterCount; ++i) {
      structures::character_selection::Character &character = characters_.emplace_back();
      stream.Read(character.refObjID);
      stream.Read(character.name);
      stream.Read(character.scale);
      stream.Read(character.curLevel);
      stream.Read(character.expOffset);
      stream.Read(character.strength);
      stream.Read(character.intelligence);
      stream.Read(character.statPoint);
      stream.Read(character.curHP);
      stream.Read(character.curMP);
      stream.Read(character.isDeleting);
      if (character.isDeleting != 0) {
        stream.Read(character.charDeleteRemainingMinutes);
      }
      stream.Read(character.guildMemberClass);
      stream.Read(character.isGuildRenameRequired);
      if (character.isGuildRenameRequired != 0) {
        stream.Read(character.currentGuildName);
      }
      stream.Read(character.academyMemberClass);
      const int itemCount = stream.Read<uint8_t>();
      for (int j=0; j<itemCount; ++j) {
        structures::character_selection::Item &item = character.items.emplace_back();
        stream.Read(item.refId);
        stream.Read(item.plus);
      }
      character.items.shrink_to_fit();
      const int avatarCount = stream.Read<uint8_t>();
      for (int j=0; j<avatarCount; ++j) {
        structures::character_selection::Avatar &avatar = character.avatarItems.emplace_back();   
        stream.Read(avatar.refId);
        stream.Read(avatar.plus);
      }
      character.avatarItems.shrink_to_fit();
    }
    characters_.shrink_to_fit();
  } else if (result_ == 2) {
    stream.Read(errorCode_);
  }
}

} // namespace packet::parsing