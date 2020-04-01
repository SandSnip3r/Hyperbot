#ifndef CONFIG_DATA_HPP_
#define CONFIG_DATA_HPP_

#include "characterLoginData.hpp"
#include "iniReader.hpp"

#include <filesystem>
#include <string>

namespace config {

class ConfigData {
public:
  ConfigData(config::IniReader &configReader);
  std::experimental::filesystem::v1::path silkroadDirectory() const;
  const CharacterLoginData& characterLoginData() const;
private:
  std::experimental::filesystem::v1::path silkroadDirectory_;
  CharacterLoginData characterLoginData_;
  void readConfig(config::IniReader &configReader);
};

} // namespace config

#endif // CONFIG_DATA_HPP_