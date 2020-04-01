#ifndef PK2_MEDIA_GAME_DATA_HPP
#define PK2_MEDIA_GAME_DATA_HPP

#include "characterData.hpp"
#include "itemData.hpp"
#include "skillData.hpp"
#include "teleportData.hpp"
#include "../../../common/pk2/divisionInfo.hpp"
#include "../../../common/pk2/pk2ReaderModern.hpp"

#include <filesystem>
#include <string>

namespace pk2 {

class GameData {
public:
  // Opens Media.PK2, parses game data into memory, and closes Media.pk2
  GameData(const std::experimental::filesystem::v1::path &kSilkroadPath);

  const DivisionInfo& divisionInfo() const;
  const CharacterData& characterData() const;
  const ItemData& itemData() const;
  const SkillData& skillData() const;
  const TeleportData& teleportData() const;
private:
  const std::experimental::filesystem::v1::path kSilkroadPath_;
  DivisionInfo divisionInfo_;
  CharacterData characterData_;
  ItemData itemData_;
  SkillData skillData_;
  TeleportData teleportData_;
  void parseMedia(Pk2ReaderModern &pk2Reader);
  void parseDivisionInfo(Pk2ReaderModern &pk2Reader);
  void parseCharacterData(Pk2ReaderModern &pk2Reader);
  void parseItemData(Pk2ReaderModern &pk2Reader);
  void parseSkillData(Pk2ReaderModern &pk2Reader);
  void parseTeleportData(Pk2ReaderModern &pk2Reader);
};

} // namespace pk2

#endif // PK2_MEDIA_GAME_DATA_HPP