#ifndef CONFIG_CHARACTER_CONFIG_HPP_
#define CONFIG_CHARACTER_CONFIG_HPP_

#include "ui-proto/character_config.pb.h"

// #include <absl/strings/string_view.h>

#include <filesystem>
// #include <optional>
// #include <string>

namespace config {

struct LoginInfo {
  std::string username;
  std::string password;
};

class CharacterConfig {
public:
  void initialize(const std::filesystem::path &pathToConfig);
  // void save();
  // void overwriteConfigProto(const proto::config::Config &configProto);
  proto::character_config::CharacterConfig& proto();
  const proto::character_config::CharacterConfig& proto() const;
  // proto::config::CharacterConfig* getCharacterConfig(absl::string_view characterName);
  // const proto::config::CharacterConfig* getCharacterConfig(absl::string_view characterName) const;
  // std::optional<LoginInfo> getLoginInfo(absl::string_view characterName) const;
private:
  static constexpr const bool kProtobufSavedAsBinary_{false};
  inline static const std::string kConfigFileFilename{"character_config"};
  // std::filesystem::path configFileFilePath_;
  proto::character_config::CharacterConfig configProto_;
};

} // namespace config

#endif // CONFIG_CHARACTER_CONFIG_HPP_