#include "session.hpp"

Session::Session(const pk2::media::GameData &gameData, const fs::path &kSilkroadDirectoryPath) :
    gameData_(gameData),
    loader_(kSilkroadDirectoryPath, gameData_.divisionInfo()) {
  //
}

Session::~Session() {
  proxy_.Stop();
  // loader_.killClient();
}

void Session::start() {
  loader_.startClient(); //throws if problem starting client & injecting
  proxy_.start(); //throws if socket issues, blocks
}