#include "hyperbot.hpp"

#include "helpers.hpp"
#include "session.hpp"
#include "state/worldState.hpp"
#include "ui/userInterface.hpp"

void Hyperbot::run() {
  parseConfig();

  ui::UserInterface userInterface{gameData_, eventBroker_};
  userInterface.initialize();

  eventBroker_.runAsync();
  userInterface.runAsync();

  initializeGameData();

  // Create a single WorldState to be shared across all sessions.
  state::WorldState worldState{gameData_, eventBroker_};

  try {
    Session session{gameData_, serverConfig_.clientPath(), eventBroker_, worldState};
    // Session session2{gameData_, serverConfig_.clientPath(), eventBroker_, worldState};
    session.initialize();
    // session2.initialize();

    // Do not call any other Session functions before the UI's WorldState is set.
    userInterface.setWorldState(worldState);

    session.setCharacterToLogin("_Nuked_");
    // session2.setCharacterToLogin("IP_Man");
    userInterface.broadcastLaunch();
    session.runAsync();
    // session2.runAsync();
    while (1) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  } catch (const std::exception &ex) {
    LOG(INFO) << "Error while running session: \"" << ex.what() << '"';
  }
}

void Hyperbot::parseConfig() {
  const auto appDataDirectory = helpers::getAppDataDirectory();
  serverConfig_.initialize(appDataDirectory);
}

void Hyperbot::initializeGameData() {
  try {
    gameData_.parseSilkroadFiles(serverConfig_.clientPath());
  } catch (const std::exception &ex) {
    throw std::runtime_error(absl::StrFormat("Failed to parse the Silkroad game data: \"%s\"", ex.what()));
  }
}