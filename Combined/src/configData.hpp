#ifndef CONFIG_DATA_HPP_
#define CONFIG_DATA_HPP_

#include "iniReader.hpp"

#include <filesystem>
#include <string>

namespace config {

class ConfigData {
public:
  ConfigData(ini::IniReader &configReader);
  std::experimental::filesystem::v1::path silkroadDirectory() const;
  std::string charName() const;
  std::string charId() const;
  std::string charPassword() const;
private:
  std::experimental::filesystem::v1::path silkroadDirectory_;
  std::string charName_;
  std::string charId_;
  std::string charPassword_;
  void readConfig(ini::IniReader &configReader);
};

} // namespace config

#endif // CONFIG_DATA_HPP_