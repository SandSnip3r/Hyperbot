#ifndef STATLOGGER_HPP_
#define STATLOGGER_HPP_

#include "broker/eventBroker.hpp"
#include "state/worldState.hpp"

#include <ui-proto/stats.pb.h>

#include <fstream>
#include <string>

// Need to open a file, subscribe to events, and stream the events to the end of the file
// Need to update a registry file which contains details about which files are for who

class StatAggregator {
public:
  StatAggregator(const state::WorldState &worldState, broker::EventBroker &eventBroker);
  void handleEvent(const event::Event *event);
private:
  static constexpr const uint32_t kVersionNum{1};
  const state::WorldState &worldState_;
  broker::EventBroker &eventBroker_;
  std::string characterName_;
  std::string filename_;
  std::ofstream statFile_;
  bool initialized_{false};
  void initialize(const std::string &characterName);
  std::string generateFilename() const;
  void writeEventToStatFile(const proto::stats::StatEvent &event);
  void printParsedFiles(const proto::stats::StatFileRegistry &registry) const;
};

#endif // STATLOGGER_HPP_