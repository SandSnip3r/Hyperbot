#include "hyperbot.hpp"

#include "clientManagerInterface.hpp"
#include "helpers.hpp"
#include "rl/rlTrainingManager.hpp"
#include "session.hpp"
#include "state/worldState.hpp"
#include "ui/userInterface.hpp"

#include <absl/log/log.h>

#include <stdexcept>

void Hyperbot::run() {
  VLOG(1) << "Running Hyperbot";
  parseConfig();

  zmq::context_t context;

  ui::UserInterface userInterface{context, gameData_, eventBroker_};
  userInterface.initialize();
  ClientManagerInterface clientManagerInterface(context);

  clientManagerInterface.runAsync();
  eventBroker_.runAsync();
  userInterface.runAsync();

  initializeGameData();

  // Create a single WorldState to be shared across all sessions.
  state::WorldState worldState{gameData_, eventBroker_};

  // Session::initialize() may be called before UserInterface::setWorldState, but not Session::runAsync().
  userInterface.setWorldState(worldState);
  userInterface.broadcastLaunch();

  constexpr bool kDoRlTraining{true};
  if constexpr (kDoRlTraining) {
    rl::RlTrainingManager rlTrainingManager{gameData_, eventBroker_, worldState, clientManagerInterface};
    rlTrainingManager.run();
  } else {
    Session session{gameData_, eventBroker_, worldState, clientManagerInterface};
    // Session session2{gameData_, eventBroker_, worldState, clientManagerInterface};
    session.initialize();
    // session2.initialize();

    // session.setCharacterToLogin("_Nuked_");
    // session2.setCharacterToLogin("IP_Man");
    session.runAsync();
    // session2.runAsync();
    VLOG(1) << "Session(s) running. Main thread now blocks.";
    while (1) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }
}

Hyperbot::~Hyperbot() {
  VLOG(1) << "Destructing Hyperbot";
}

void Hyperbot::parseConfig() {
  const auto appDataDirectory = helpers::getAppDataDirectory();
  VLOG(2) << absl::StreamFormat("Parsing Hyperbot config at \"%s\"", appDataDirectory);
  serverConfig_.initialize(appDataDirectory);
  VLOG(2) << "Finished parsing config";
}

void Hyperbot::initializeGameData() {
  try {
    gameData_.parseSilkroadFiles(serverConfig_.clientPath());
  } catch (const std::exception &ex) {
    throw std::runtime_error(absl::StrFormat("Failed to parse the Silkroad game data: \"%s\"", ex.what()));
  }
}