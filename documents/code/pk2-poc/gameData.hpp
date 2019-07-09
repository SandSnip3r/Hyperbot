#ifndef GAME_DATA_HPP
#define GAME_DATA_HPP

#include "itemData.hpp"
#include "skillData.hpp"

#include <string>

class GameData {
public:
  // Opens the necessary PK2 files and parses game data into memory
  // clientDirectory contains things like sro_client.exe, Media.pk2, Music.pk2, etc.
  // throws if filesystem errors occur
  GameData(const std::string &clientDirectory);

  // Closes any open PK2 files
  ~GameData();

  std::string gatewayAddress() const;
  const ItemData& itemData() const;
  const SkillData& skillData() const;
private:
  void parseMedia(const std::string &clientDirectory);
  void initializeItemData();
  void initializeSkillData();
  PK2Reader pk2Reader_;
  ItemData itemData_;
  SkillData skillData_;
};

#endif // GAME_DATA_HPP