#ifndef GAME_DATA_HPP
#define GAME_DATA_HPP

#include "itemData.hpp"
#include "skillData.hpp"
#include "../../common/divisionInfo.hpp"
#include "../../common/pk2ReaderModern.hpp"

#include <filesystem>
#include <string>

namespace pk2::media {

class GameData {
public:
  // Opens Media.PK2, parses game data into memory, and closes Media.pk2
  GameData(const std::experimental::filesystem::v1::path &kSilkroadPath);

  // Returns a const reference to division info
  const pk2::DivisionInfo& divisionInfo() const;

  // Returns a const reference to item data
  const ItemData& itemData() const;

  // Returns a const reference to skill data
  const SkillData& skillData() const;
private:
  const std::experimental::filesystem::v1::path kSilkroadPath_;
  pk2::DivisionInfo divisionInfo_;
  ItemData itemData_;
  SkillData skillData_;
  void parseMedia(pk2::Pk2ReaderModern &pk2Reader);
  void parseDivisionInfo(pk2::Pk2ReaderModern &pk2Reader);
  void parseItemData(pk2::Pk2ReaderModern &pk2Reader);
  void parseSkillData(pk2::Pk2ReaderModern &pk2Reader);
};

} // namespace pk2::media

#endif // GAME_DATA_HPP