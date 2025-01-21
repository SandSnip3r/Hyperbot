#ifndef CONFIG_HPP_
#define CONFIG_HPP_

#include <ui_proto/rl_ui_config.pb.h>

#include <filesystem>
#include <string_view>

class Config {
public:
  Config(const std::filesystem::path &filePath);
  void load();
  void save() const;
  proto::rl_ui_config::Config& proto() { return proto_; }
private:
  static constexpr std::string_view kDefaultIpAddress_{"127.0.0.1"};
  static constexpr int kDefaultPort_{5555};
  static constexpr bool kProtobufSavedAsBinary_{false};
  const std::filesystem::path &filePath_;
  proto::rl_ui_config::Config proto_;

  void makeAndSaveDefaultConfig();
};

#endif // CONFIG_HPP_
