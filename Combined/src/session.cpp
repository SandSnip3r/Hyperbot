#include "session.hpp"

Session::Session(const std::string &silkroadPath, uint8_t locale) :
    kSilkroadPath_(silkroadPath),
    kLocale_(locale),
    loader_(kSilkroadPath_, kLocale_) {
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