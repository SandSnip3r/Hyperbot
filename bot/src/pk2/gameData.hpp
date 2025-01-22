#ifndef PK2_MEDIA_GAME_DATA_HPP
#define PK2_MEDIA_GAME_DATA_HPP

#include "characterData.hpp"
#include "itemData.hpp"
#include "levelData.hpp"
#include "magicOptionData.hpp"
#include "masteryData.hpp"
#include "refRegion.hpp"
#include "shopData.hpp"
#include "skillData.hpp"
#include "teleportData.hpp"
#include "textData.hpp"
#include "textZoneNameData.hpp"

#include <silkroad_lib/navmesh/navmesh.hpp>
#include <silkroad_lib/navmesh/triangulation/navmeshTriangulation.hpp>
#include <silkroad_lib/pk2/divisionInfo.hpp>
#include <silkroad_lib/pk2/pk2ReaderModern.hpp>
#include <silkroad_lib/pk2/regionInfo.hpp>

#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

namespace pk2 {

class GameData {
public:
  // Opens Media.PK2 and Data.PK2, parses game data into memory, then closes Media.pk2 and Data.PK2
  void parseSilkroadFiles(const std::filesystem::path &clientPath);
  const uint16_t gatewayPort() const;
  const sro::pk2::DivisionInfo& divisionInfo() const;
  const CharacterData& characterData() const;
  const ItemData& itemData() const;
  const ShopData& shopData() const;
  const SkillData& skillData() const;
  const MasteryData& masteryData() const;
  const MagicOptionData& magicOptionData() const;
  const LevelData& levelData() const;
  const RefRegion& refRegion() const;
  const TextData& textData() const;
  const TextZoneNameData& textZoneNameData() const;
  const TeleportData& teleportData() const;
  const sro::navmesh::triangulation::NavmeshTriangulation& navmeshTriangulation() const;
  const sro::pk2::RegionInfo& regionInfo() const;

  std::string getSkillName(sro::scalar_types::ReferenceObjectId skillRefId) const;
  std::string getItemName(sro::scalar_types::ReferenceObjectId itemRefId) const;
  std::string getMasteryName(sro::pk2::ref::MasteryId masteryId) const;

  // Expecting names like Pacheon, WIZARD, roGue, fiRE, etc. Case does not matter.
  sro::pk2::ref::MasteryId getMasteryId(std::string masteryName) const;
private:
  std::mutex printMutex_;
  uint16_t gatewayPort_;
  sro::pk2::DivisionInfo divisionInfo_;
  CharacterData characterData_;
  ItemData itemData_;
  ShopData shopData_;
  SkillData skillData_;
  MasteryData masteryData_;
  MagicOptionData magicOptionData_;
  LevelData levelData_;
  RefRegion refRegion_;
  TextData textData_;
  TextZoneNameData textZoneNameData_;
  TeleportData teleportData_;

  std::optional<sro::navmesh::Navmesh> navmesh_;
  std::optional<sro::navmesh::triangulation::NavmeshTriangulation> navmeshTriangulation_;

  sro::pk2::RegionInfo regionInfo_;

  void parseData(sro::pk2::Pk2ReaderModern &pk2Reader);
  void parseNavmeshData(sro::pk2::Pk2ReaderModern &pk2Reader);
  void parseRegionInfo(sro::pk2::Pk2ReaderModern &pk2Reader);

  void parseMedia(sro::pk2::Pk2ReaderModern &pk2Reader);
  void parseGatewayPort(sro::pk2::Pk2ReaderModern &pk2Reader);
  void parseDivisionInfo(sro::pk2::Pk2ReaderModern &pk2Reader);
  void parseCharacterData(sro::pk2::Pk2ReaderModern &pk2Reader);
  void parseItemData(sro::pk2::Pk2ReaderModern &pk2Reader);
  void parseSkillData(sro::pk2::Pk2ReaderModern &pk2Reader);
  void parseMasteryData(sro::pk2::Pk2ReaderModern &pk2Reader);
  void parseTeleportData(sro::pk2::Pk2ReaderModern &pk2Reader);
  void parseShopData(sro::pk2::Pk2ReaderModern &pk2Reader);
  void parseMagicOptionData(sro::pk2::Pk2ReaderModern &pk2Reader);
  void parseLevelData(sro::pk2::Pk2ReaderModern &pk2Reader);
  void parseRefRegion(sro::pk2::Pk2ReaderModern &pk2Reader);
  void parseTextData(sro::pk2::Pk2ReaderModern &pk2Reader);
  void parseTextZoneName(sro::pk2::Pk2ReaderModern &pk2Reader);
  void parseText(sro::pk2::Pk2ReaderModern &pk2Reader);
  void parseTextUiSystem(sro::pk2::Pk2ReaderModern &pk2Reader);
};

} // namespace pk2

#endif // PK2_MEDIA_GAME_DATA_HPP