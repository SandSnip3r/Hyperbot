#ifndef PK2_MEDIA_GAME_DATA_HPP
#define PK2_MEDIA_GAME_DATA_HPP

#include "characterData.hpp"
#include "itemData.hpp"
#include "shopData.hpp"
#include "skillData.hpp"
#include "teleportData.hpp"
#include "../navmesh/navmeshParser.hpp" // TODO: Maybe move to the common parsing area?
#include "../../../common/pk2/divisionInfo.hpp"
#include "../../../common/pk2/pk2ReaderModern.hpp"

#include "triangle/triangle.h"

#include <filesystem>
#include <mutex>
#include <string>

namespace pk2 {

class GameData {
public:
  // Opens Media.PK2, parses game data into memory, and closes Media.pk2
  GameData(const std::filesystem::path &kSilkroadPath);

  const DivisionInfo& divisionInfo() const;
  const CharacterData& characterData() const;
  const ItemData& itemData() const;
  const ShopData& shopData() const;
  const SkillData& skillData() const;
  const TeleportData& teleportData() const;
  const triangle::triangleio& getSavedTriangleData() const;
  const triangle::triangleio& getSavedTriangleVoronoiData() const;
private:
  std::mutex printMutex_;
  const std::filesystem::path kSilkroadPath_;
  DivisionInfo divisionInfo_;
  CharacterData characterData_;
  ItemData itemData_;
  ShopData shopData_;
  SkillData skillData_;
  TeleportData teleportData_;

  // TODO: Remove
	triangle::triangleio savedTriangleData_;
	triangle::triangleio savedTriangleVoronoiData_;

  void parseData(Pk2ReaderModern &pk2Reader);
  void parseNavmeshData(Pk2ReaderModern &pk2Reader);

  void parseMedia(Pk2ReaderModern &pk2Reader);
  void parseDivisionInfo(Pk2ReaderModern &pk2Reader);
  void parseCharacterData(Pk2ReaderModern &pk2Reader);
  void parseItemData(Pk2ReaderModern &pk2Reader);
  void parseSkillData(Pk2ReaderModern &pk2Reader);
  void parseTeleportData(Pk2ReaderModern &pk2Reader);
  void parseShopData(Pk2ReaderModern &pk2Reader);

  void buildTriangleDataForRegion(const RegionNavmesh &regionNavmesh, const NavmeshParser &navmeshParser);
};

} // namespace pk2

#endif // PK2_MEDIA_GAME_DATA_HPP