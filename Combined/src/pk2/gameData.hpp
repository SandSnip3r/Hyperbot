#ifndef PK2_MEDIA_GAME_DATA_HPP
#define PK2_MEDIA_GAME_DATA_HPP

#include "characterData.hpp"
#include "itemData.hpp"
#include "levelData.hpp"
#include "magicOptionData.hpp"
#include "shopData.hpp"
#include "skillData.hpp"
#include "teleportData.hpp"
#include "textItemAndSkillData.hpp"
#include "textZoneNameData.hpp"
#include "navmesh/navmesh.hpp"
#include "navmesh/triangulation/navmeshTriangulation.hpp"
#include "../../../common/pk2/divisionInfo.hpp"
#include "../../../common/pk2/pk2ReaderModern.hpp"

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
  const DivisionInfo& divisionInfo() const;
  const CharacterData& characterData() const;
  const ItemData& itemData() const;
  const ShopData& shopData() const;
  const SkillData& skillData() const;
  const MagicOptionData& magicOptionData() const;
  const LevelData& levelData() const;
  const TextItemAndSkillData& textItemAndSkillData() const;
  const TextZoneNameData& textZoneNameData() const;
  const TeleportData& teleportData() const;
  const navmesh::triangulation::NavmeshTriangulation& navmeshTriangulation() const;

  std::optional<std::string> getSkillNameIfExists(sro::scalar_types::ReferenceObjectId skillRefId) const;
private:
  std::mutex printMutex_;
  uint16_t gatewayPort_;
  DivisionInfo divisionInfo_;
  CharacterData characterData_;
  ItemData itemData_;
  ShopData shopData_;
  SkillData skillData_;
  MagicOptionData magicOptionData_;
  LevelData levelData_;
  TextItemAndSkillData textItemAndSkillData_;
  TextZoneNameData textZoneNameData_;
  TeleportData teleportData_;
  std::optional<navmesh::Navmesh> navmesh_;
  std::optional<navmesh::triangulation::NavmeshTriangulation> navmeshTriangulation_;

  void parseData(Pk2ReaderModern &pk2Reader);
  void parseNavmeshData(Pk2ReaderModern &pk2Reader);

  void parseMedia(Pk2ReaderModern &pk2Reader);
  void parseGatewayPort(Pk2ReaderModern &pk2Reader);
  void parseDivisionInfo(Pk2ReaderModern &pk2Reader);
  void parseCharacterData(Pk2ReaderModern &pk2Reader);
  void parseItemData(Pk2ReaderModern &pk2Reader);
  void parseSkillData(Pk2ReaderModern &pk2Reader);
  void parseTeleportData(Pk2ReaderModern &pk2Reader);
  void parseShopData(Pk2ReaderModern &pk2Reader);
  void parseMagicOptionData(Pk2ReaderModern &pk2Reader);
  void parseLevelData(Pk2ReaderModern &pk2Reader);
  void parseTextData(Pk2ReaderModern &pk2Reader);
  void parseTextZoneName(Pk2ReaderModern &pk2Reader);
  void parseTextItemAndSkill(Pk2ReaderModern &pk2Reader);
};

} // namespace pk2

#endif // PK2_MEDIA_GAME_DATA_HPP