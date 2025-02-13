#include "session.hpp"

Session::Session(const pk2::GameData &gameData,
                 broker::EventBroker &eventBroker,
                 state::WorldState &worldState,
                 ClientManagerInterface &clientManagerInterface) :
    gameData_(gameData),
    eventBroker_(eventBroker),
    bot_(sessionId_, gameData_, proxy_, packetBroker_, eventBroker_, worldState),
    clientManagerInterface_(clientManagerInterface) {
}

void Session::setCharacter(const CharacterLoginInfo &characterLoginInfo) {
  bot_.setCharacter(characterLoginInfo);
}

void Session::initialize() {
  if (initialized_) {
    throw std::runtime_error("Session::initialize already initialized");
  }
  bot_.initialize();
  initialized_ = true;
}

void Session::runAsync() {
  if (!initialized_) {
    throw std::runtime_error("Session::runAsync called before Session::initialize");
  }
  proxy_.runAsync();
}

const state::WorldState& Session::getWorldState() const {
  return bot_.worldState();
}

Bot& Session::getBot() {
  return bot_;
}

std::future<void> Session::asyncOpenClient() {
  clientId_ = clientManagerInterface_.startClient(proxy_.getOurListeningPort());
  return bot_.asyncOpenClient();
}

std::atomic<SessionId> Session::nextSessionId{0};

SessionId Session::createUniqueSessionId() {
  return nextSessionId.fetch_add(1);
}