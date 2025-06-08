#include "session.hpp"

Session::Session(const pk2::GameData &gameData,
                 broker::EventBroker &eventBroker,
                 state::WorldState &worldState,
                 ClientManagerInterface &clientManagerInterface,
                 ui::RlUserInterface &rlUserInterface) :
    gameData_(gameData),
    eventBroker_(eventBroker),
    worldState_(worldState),
    rlUserInterface_(rlUserInterface),
    clientManagerInterface_(clientManagerInterface) {
  eventBroker_.subscribeToEvent(event::EventCode::kClientDied, std::bind(&Session::handleClientDiedEvent, this, std::placeholders::_1));
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
  return worldState_;
}

Bot& Session::getBot() {
  return bot_;
}

std::future<void> Session::asyncOpenClient() {
  clientId_ = clientManagerInterface_.startClient(proxy_.getOurListeningPort());
  return bot_.getReadyToLoginFuture();
}

std::future<void> Session::connectClientlessAsync() {
  std::future<void> future = bot_.getReadyToLoginFuture();
  proxy_.connectClientlessAsync();
  return future;
}

void Session::handleClientDiedEvent(const event::Event *event) {
  const auto &clientDiedEvent = dynamic_cast<const event::ClientDied&>(*event);
  if (clientId_ && clientDiedEvent.clientId == *clientId_) {
    VLOG(1) << absl::StreamFormat("Our client died! Session %d, client %d", sessionId_, *clientId_);
    proxy_.setClientless(true);
    clientId_.reset();
    // TODO: We should unsubscribe from this event, because it'll never trigger again.
    //   However, there is a bug in EventBroker. We cannot unsubscribe from an event that is currently being handled.
  }
}

std::atomic<SessionId> Session::nextSessionId{0};

SessionId Session::createUniqueSessionId() {
  return nextSessionId.fetch_add(1);
}