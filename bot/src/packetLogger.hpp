#ifndef PACKET_LOGGER_HPP_
#define PACKET_LOGGER_HPP_

#include "packet/opcode.hpp"
#include "shared/silkroad_security.h"
#include "shared/stream_utility.h"

#include <fstream>
#include <string>
#include <vector>

class PacketLogger {
public:
  PacketLogger(const std::string &logDirectoryPath);
  void logPacket(const PacketContainer &packet, bool blocked, PacketContainer::Direction direction);
private:
  static const bool kLogToFile;
  static const int kLogToConsoleMinimumVlogLevel;
  const std::string logFileDirectoryPath_;
  const std::string logFilePath_;
  std::ofstream logfile_;
  inline static const std::vector<packet::Opcode> opcodeConsoleLoggingWhitelist_ = {};
  void logPacketToFile(int64_t msSinceEpoch, const PacketContainer &packet, bool blocked, PacketContainer::Direction direction);
  void logPacketToConsole(int64_t msSinceEpoch, const PacketContainer &packet, bool blocked, PacketContainer::Direction direction);
  std::string getLogFilePath() const;
};

#endif // PACKET_LOGGER_HPP_