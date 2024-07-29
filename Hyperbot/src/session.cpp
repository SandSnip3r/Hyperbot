#include "session.hpp"

Session::Session(const pk2::GameData &gameData,
                 std::string_view clientPath,
                 broker::EventBroker &eventBroker) :
    gameData_(gameData),
    eventBroker_(eventBroker),
    loader_(clientPath, gameData_.divisionInfo()) {
}

Session::~Session() {
  proxy_.stop();
  proxyThread_.join();
  // loader_.killClient();
}

void Session::setCharacterToLogin(std::string_view characterName) {
  bot_.setCharacterToLogin(characterName);
}

void Session::initialize() {
  bot_.initialize();
}

void Session::runAsync() {
  loader_.startClient(proxy_.getOurListeningPort()); //throws if problem starting client & injecting
  proxyThread_ = std::thread(std::bind(&Proxy::run, &proxy_));
  // proxy_.run(); //throws if socket issues, blocks
}

const state::WorldState& Session::getWorldState() const {
  return bot_.worldState();
}

std::atomic<SessionId> Session::nextSessionId{0};

SessionId Session::createUniqueSessionId() {
  return nextSessionId.fetch_add(1);
}