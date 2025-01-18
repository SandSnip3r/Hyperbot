#ifndef CONFIG_SERVER_CONFIG_HPP_
#define CONFIG_SERVER_CONFIG_HPP_

#include <ui_proto/server_config.pb.h>

// #include <absl/strings/string_view.h>

#include <filesystem>
// #include <optional>
#include <string>
#include <string_view>

namespace config {

class ServerConfig {
public:
  void initialize(const std::filesystem::path &pathToConfig);
  // void save();
  // void overwriteConfigProto(const proto::config::Config &configProto);
  // proto::config::Config& configProto();
  // const proto::config::Config& configProto() const;
  // proto::config::CharacterConfig* getCharacterConfig(absl::string_view characterName);
  // const proto::config::CharacterConfig* getCharacterConfig(absl::string_view characterName) const;
  // std::optional<LoginInfo> getLoginInfo(absl::string_view characterName) const;
  std::string_view clientPath() const { return clientPath_; }
private:
  static constexpr const bool kProtobufSavedAsBinary_{false};
  inline static const std::string kConfigFileFilename{"server_config"};
  // std::filesystem::path configFileFilePath_;
  // proto::config::Config configProto_;
  std::string clientPath_;
};

} // namespace config

#endif // CONFIG_SERVER_CONFIG_HPP_