#ifndef CONFIG_CONFIG_HPP_
#define CONFIG_CONFIG_HPP_

#include "ui-proto/config.pb.h"

#include <absl/strings/string_view.h>

#include <filesystem>
#include <optional>
#include <string>

namespace config {

struct LoginInfo {
  std::string username;
  std::string password;
};

class Config {
public:
  void initialize(const std::filesystem::path &pathToConfig);
  void save();
  void overwriteConfigProto(const proto::config::Config &configProto);
  proto::config::Config& configProto();
  const proto::config::Config& configProto() const;
  proto::config::CharacterConfig* getCharacterConfig(absl::string_view characterName);
  const proto::config::CharacterConfig* getCharacterConfig(absl::string_view characterName) const;
  std::optional<LoginInfo> getLoginInfo(absl::string_view characterName) const;
private:
  static constexpr const bool kProtobufSavedAsBinary_{false};
  inline static const std::string kConfigFileFilename{"config"};
  std::filesystem::path configFileFilePath_;
  proto::config::Config configProto_;
};

} // namespace config

#endif // CONFIG_CONFIG_HPP_