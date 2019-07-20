#include "configData.hpp"

#include <algorithm>
#include <string>

namespace config {

ConfigData::ConfigData(ini::IniReader &configReader) {
  readConfig(configReader);
}

std::experimental::filesystem::v1::path ConfigData::silkroadDirectory() const {
  return silkroadDirectory_;
}

std::string ConfigData::charName() const {
  return charName_;
}

std::string ConfigData::charId() const {
  return charId_;
}

std::string ConfigData::charPassword() const {
  return charPassword_;
}

void ConfigData::readConfig(ini::IniReader &configReader) {
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

  charName_ = sections[0];
  if (!configReader.valueExists(charName_, "id")) {
    throw std::runtime_error("Missing \"id\" entry for character \""+charName_+"\"");
  }
  if (!configReader.valueExists(charName_, "password")) {
    throw std::runtime_error("Missing \"password\" entry for character \""+charName_+"\"");
  }
  
  charId_ = configReader.get<std::string>(charName_, "id");
  charPassword_ = configReader.get<std::string>(charName_, "password");
}

} // namespace config