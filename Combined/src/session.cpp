#include "session.hpp"

Session::Session(const std::string &configPath) : kConfigPath_(configPath) {
  //Load config
  // Config config(kConfigPath_); //throws if file not found or cant open
  // bot_.configure(config);
  // proxy_.configure(config);
}

Session::~Session() {
  proxy_.Stop();
  loader_.killClient();
}

void Session::start() {
  loader_.startClient(); //throws if problem starting client & injecting
  proxy_.start(); //throws if socket issues, blocks
}