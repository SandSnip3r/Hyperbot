#include "configData.hpp"

#include <algorithm>
#include <string>

namespace config {

ConfigData::ConfigData(config::IniReader &configReader) {
  readConfig(configReader);
}

std::filesystem::path ConfigData::silkroadDirectory() const {
  return silkroadDirectory_;
}

const CharacterLoginData& ConfigData::characterLoginData() const {
  return characterLoginData_;
}

void ConfigData::readConfig(config::IniReader &configReader) {
  const std::string kSilkroadDataSectionName = "Silkroad";
  
  if (!configReader.valueExists(kSilkroadDataSectionName, "path")) {
    throw std::runtime_error("Config file missing \"path\" entry in \"Silkroad\" section");
  }
  silkroadDirectory_ = configReader.get<std::string>(kSilkroadDataSectionName, "path");

  auto sections = configReader.getSections();
  sections.erase(std::remove(sections.begin(), sections.end(), kSilkroadDataSectionName), sections.end());

  if (sections.empty()) {
    throw std::runtime_error("Unable to find section for character");
  }

  characterLoginData_.name = sections[0];
  if (!configReader.valueExists(characterLoginData_.name, "id")) {
    throw std::runtime_error("Missing \"id\" entry for character \""+characterLoginData_.name+"\"");
  }
  if (!configReader.valueExists(characterLoginData_.name, "password")) {
    throw std::runtime_error("Missing \"password\" entry for character \""+characterLoginData_.name+"\"");
  }
  
  characterLoginData_.id = configReader.get<std::string>(characterLoginData_.name, "id");
  characterLoginData_.password = configReader.get<std::string>(characterLoginData_.name, "password");
}

} // namespace config