#ifndef PK2_MEDIA_GAME_DATA_HPP
#define PK2_MEDIA_GAME_DATA_HPP

#include "characterData.hpp"
#include "itemData.hpp"
#include "skillData.hpp"
#include "teleportData.hpp"
#include "../../common/divisionInfo.hpp"
#include "../../common/pk2ReaderModern.hpp"

#include <filesystem>
#include <string>

namespace pk2::media {

class GameData {
public:
  // Opens Media.PK2, parses game data into memory, and closes Media.pk2
  GameData(const std::experimental::filesystem::v1::path &kSilkroadPath);

  const pk2::DivisionInfo& divisionInfo() const;
  const CharacterData& characterData() const;
  const ItemData& itemData() const;
  const SkillData& skillData() const;
  const TeleportData& teleportData() const;
private:
  const std::experimental::filesystem::v1::path kSilkroadPath_;
  pk2::DivisionInfo divisionInfo_;
  CharacterData characterData_;
  ItemData itemData_;
  SkillData skillData_;
  TeleportData teleportData_;
  void parseMedia(pk2::Pk2ReaderModern &pk2Reader);
  void parseDivisionInfo(pk2::Pk2ReaderModern &pk2Reader);
  void parseCharacterData(pk2::Pk2ReaderModern &pk2Reader);
  void parseItemData(pk2::Pk2ReaderModern &pk2Reader);
  void parseSkillData(pk2::Pk2ReaderModern &pk2Reader);
  void parseTeleportData(pk2::Pk2ReaderModern &pk2Reader);
};

} // namespace pk2::media

#endif // PK2_MEDIA_GAME_DATA_HPP