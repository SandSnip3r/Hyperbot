#include "session.hpp"
#include "silkroadConnection.hpp"

#include "broker/eventBroker.hpp"
#include "config/configData.hpp"
#include "config/iniReader.hpp"
#include "pk2/gameData.hpp"
#include "ui/userInterface.hpp"
#include "ui-proto/config.pb.h"

#include "../../common/Common.h"
#include "../../common/pk2/divisionInfo.hpp"
#include "../../common/pk2/parsing/parsing.hpp"

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/log/globals.h>
#include <absl/log/initialize.h>
#include <absl/log/log.h>
#include <absl/strings/str_format.h>

#include <charconv>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

constexpr std::string_view kDefaultCharacterFlagValue{"__default_character__"};

ABSL_FLAG(int, v, -1, "Verbose logging level");
ABSL_FLAG(std::vector<std::string>, vmodule, {}, "Per-module verbose logging level");
ABSL_FLAG(std::string, character, std::string(kDefaultCharacterFlagValue), "Character to login");

using namespace std;
namespace fs = std::filesystem;

fs::path getAppDataDirectory();

class Hyperbot {
public:
  Hyperbot(const std::optional<std::string> &characterToLogin) : characterToLogin_(characterToLogin) {}
  void run() {
    const auto appDataDirectory = getAppDataDirectory();
    config_.initialize(appDataDirectory);
    if (characterToLogin_) {
      // TODO: This will result in overwriting of the original config file if this is saved.
      LOG(INFO) << "Given a character to log into: \"" << *characterToLogin_ << "\". Overwritting config.";
      config_.configProto().set_character_to_login(*characterToLogin_);
    } else {
      LOG(INFO) << "Using default character to login";
    }

    subscribeToEvents();
    ui::UserInterface userInterface{gameData_, eventBroker_}; // TODO: Remove gameData_ from user interface
    userInterface.initialize();

    eventBroker_.runAsync();
    userInterface.runAsync();

    tryInitializeGameData();
    {
      std::unique_lock<std::mutex> gameDataLock(gameDataMutex_);
      if (!gameDataReady_) {
        // Game data hasn't yet been parsed, need to wait until the config is updated to contain a sro_client path which has parsable game data
        LOG(INFO) << "Waiting until game data is parsed";
        gameDataParsedConditionVariable_.wait(gameDataLock, [this](){
          return gameDataReady_;
        });
        LOG(INFO) << "Done parsing game data";
      }
    }

    // We were able to parse our game data; no longer want to accept new configs here.
    LOG(INFO) << "Unsubscribing from new config received events";
    eventBroker_.unsubscribeFromEvent(configReceivedSubscriptionId_);

    try {
      Session session{gameData_, config_, eventBroker_};
      session.initialize();
      userInterface.setWorldState(session.getWorldState());
      userInterface.broadcastLaunch();
      userInterface.broadcastConfig(config_.configProto());
      session.run();
    } catch (const std::exception &ex) {
      LOG(INFO) << "Error while running session: \"" << ex.what() << '"';
    }
  }

private:
  std::optional<std::string> characterToLogin_;
  config::Config config_;
  std::mutex configMutex_;
  broker::EventBroker eventBroker_;
  bool waitingForClientPath_{true};
  pk2::GameData gameData_;
  bool gameDataReady_{false};
  std::condition_variable gameDataParsedConditionVariable_;
  std::mutex gameDataMutex_;
  broker::EventBroker::SubscriptionId configReceivedSubscriptionId_;

  void subscribeToEvents() {
    auto eventHandleFunction = std::bind(&Hyperbot::handleEvent, this, std::placeholders::_1);
    configReceivedSubscriptionId_ = eventBroker_.subscribeToEvent(event::EventCode::kNewConfigReceived, eventHandleFunction);
  }

  void handleEvent(const event::Event *event) {
    if (event->eventCode == event::EventCode::kNewConfigReceived) {
      const auto &castedEvent = dynamic_cast<const event::NewConfigReceived&>(*event);
      handleNewConfigReceived(castedEvent);
    }
  }

  void handleNewConfigReceived(const event::NewConfigReceived &event) {
    {
      std::unique_lock<std::mutex> configLock(configMutex_);
      config_.overwriteConfigProto(event.config);
    }
    eventBroker_.publishEvent(event::EventCode::kConfigUpdated);

    bool gameDataIsAlreadyParsed{false};
    {
      std::unique_lock<std::mutex> gameDataLock(gameDataMutex_);
      gameDataIsAlreadyParsed = gameDataReady_;
    }
    if (gameDataIsAlreadyParsed) {
      // We do not support changing the sro_client path once we've already successfully parsed the game data
      return;
    }
    tryInitializeGameData();
  }

  void tryInitializeGameData() {
    std::string clientPath;
    {
      std::unique_lock<std::mutex> configLock(configMutex_);
      if (!config_.configProto().has_client_path()) {
        LOG(INFO) << "Config does not contain a client path. Nothing to do";
        return;
      }
      clientPath = config_.configProto().client_path();
    }
    try {
      std::unique_lock<std::mutex> gameDataLock(gameDataMutex_);

      // Overwrite old one, just in case we already tried to parse one and it failed
      // gameData_ = pk2::GameData(); // TODO: Modify GameData so that it is in a good state if parsing fails

      gameData_.parseSilkroadFiles(clientPath);
      gameDataReady_ = true;
      gameDataParsedConditionVariable_.notify_one();
    } catch (const std::exception &ex) {
      LOG(INFO) << "Failed to parse Silkroad game data: \"" << ex.what() << "\"";
      // If we fail to parse the game data, we don't know that the GameData object is in a good state
      //  Throw an error so we don't end up here again
      throw std::runtime_error("We failed to parse the Silkroad game data");
    }
  }
};

void initializeLogging() {
  absl::InitializeLog();

  // Set log levels from flags.
  absl::SetGlobalVLogLevel(absl::GetFlag(FLAGS_v));

  // Set per-module vlog levels.
  const auto vmodules = absl::GetFlag(FLAGS_vmodule);
  for (std::string_view module : vmodules) {
    const std::string::size_type equals_pos = module.find('=');
    if (equals_pos != std::string::npos) {
      std::string_view module_name = module.substr(0, equals_pos);
      std::string_view module_level = module.substr(equals_pos + 1);
      int level;
      const auto parseResult = std::from_chars(module_level.data(), module_level.data()+module_level.size(), level);
      if (parseResult.ec != std::errc{}) {
        const auto errorCode = std::make_error_code(parseResult.ec);
        throw std::runtime_error(absl::StrFormat("Could not parse integer from vmodule flag: \"%s\". Error is \"%s\"", module, errorCode.message()));
      }
      absl::SetVLogLevel(module_name, level);
    }
  }

  // Set everything to log to stderr.
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
}

int main(int argc, char **argv) {
  absl::ParseCommandLine(argc, argv);
  initializeLogging();

  std::optional<std::string> characterToLogin;
  if (absl::GetFlag(FLAGS_character) != (kDefaultCharacterFlagValue)) {
    characterToLogin = absl::GetFlag(FLAGS_character);
  }
  try {
    Hyperbot hyperbot(characterToLogin);
    hyperbot.run();
  } catch (const std::exception &ex) {
    LOG(INFO) << "Caught an exception; exiting. \"" << ex.what() << '"';
    return 1;
  }
  return 0;
}

fs::path getAppDataDirectory() {
  const auto appDataDirectoryPath = getAppDataPath();
  if (appDataDirectoryPath.empty()) {
    throw std::runtime_error("Unable to find %APPDATA%\n");
  }
  // Make sure the directory exists
  if (!fs::exists(appDataDirectoryPath)) {
    fs::create_directory(appDataDirectoryPath);
  }
  return appDataDirectoryPath;
}