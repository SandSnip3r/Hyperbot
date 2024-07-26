#include "packetLogger.hpp"

#include "packet/opcode.hpp"

#include <absl/strings/str_format.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace {

bool isPrintable(uint8_t data) {
  return (data >= ' ' && data <= '~');
}

int64_t getMsSinceEpoch() {
  const std::chrono::time_point<std::chrono::system_clock> nowTimePoint = std::chrono::system_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(nowTimePoint.time_since_epoch()).count();
}

} // anonymous namespace

const bool PacketLogger::kLogToFile = false;
const bool PacketLogger::kLogToConsole = false;

PacketLogger::PacketLogger(const std::string &logDirectoryPath) : logFileDirectoryPath_(logDirectoryPath), logFilePath_(getLogFilePath()) {
  if (kLogToFile) {
    logfile_.open(logFilePath_);
    if (!logfile_) {
      throw std::runtime_error("Unable to initialize logfile_ \""+logFilePath_+"\"");
    }
  }
}

void PacketLogger::logPacket(const PacketContainer &packet, bool blocked, PacketContainer::Direction direction) {
  int64_t msSinceEpoch = getMsSinceEpoch();
  if (kLogToFile) {
    logPacketToFile(msSinceEpoch, packet, blocked, direction);
  }
  if (kLogToConsole || (std::find(opcodeConsoleLoggingWhitelist_.begin(), opcodeConsoleLoggingWhitelist_.end(), static_cast<packet::Opcode>(packet.opcode)) != opcodeConsoleLoggingWhitelist_.end())) {
    logPacketToConsole(msSinceEpoch, packet, blocked, direction);
  }
}

void PacketLogger::logPacketToFile(int64_t msSinceEpoch, const PacketContainer &packet, bool blocked, PacketContainer::Direction direction) {
  if (!logfile_) {
    throw std::runtime_error("Log file \""+logFilePath_+"\" problem");
  }
  std::stringstream ss;
  ss << msSinceEpoch << ',';
  ss << blocked << ',';
  ss << (int)packet.encrypted << ',';
  ss << (int)packet.massive << ',';
  if (direction == PacketContainer::Direction::kClientToServer) {
    ss << "C,";
  } else if (direction == PacketContainer::Direction::kServerToClient) {
    ss << "S,";
  } else if (direction == PacketContainer::Direction::kBotToServer) {
    ss << "B,";
  } else if (direction == PacketContainer::Direction::kBotToClient) {
    ss << "b,";
  }
  ss << (int)packet.opcode;
  StreamUtility stream = packet.data;
  int byteCount = stream.GetStreamSize();
  for (int i=0; i<byteCount; ++i) {
    ss << ',' << (int)stream.Read<uint8_t>();
  }
  ss << '\n';
  logfile_ << ss.str() << std::flush;
}

void PacketLogger::logPacketToConsole(int64_t msSinceEpoch, const PacketContainer &packet, bool blocked, PacketContainer::Direction direction) {
  const int kBytesPerLine{20};

  std::stringstream ss;
  ss << '[' << msSinceEpoch << "] ";
  if (direction == PacketContainer::Direction::kClientToServer) {
    ss << " (C->S)";
  } else if (direction == PacketContainer::Direction::kServerToClient) {
    ss << " (S->C)";
  } else if (direction == PacketContainer::Direction::kBotToServer) {
    ss << " (B->S)";
  } else if (direction == PacketContainer::Direction::kBotToClient) {
    ss << " (B->C)";
  }
  ss << (int)blocked << ',';
  ss << (int)packet.encrypted << ',';
  ss << (int)packet.massive << ',';
  ss << std::hex << packet.opcode << std::dec << ' ';
  const int indentSize = ss.str().size();
  StreamUtility stream = packet.data;
  const auto &dataVector = stream.GetStreamVector();
  std::vector<uint8_t> copiedData(dataVector.begin(), dataVector.end());
  const int lineCount = std::ceil((float)copiedData.size()/kBytesPerLine);
  for (int lineNum=0; lineNum<lineCount; ++lineNum) {
    const int startingIndex = kBytesPerLine*lineNum;
    for (int i=startingIndex; ((i < (startingIndex + kBytesPerLine)) && (i < copiedData.size())); ++i) {
      ss << std::setfill('0') << std::setw(2) << std::hex << (int)copiedData.at(i) << std::dec << ' ';
    }
    ss << '\n';
    if (lineNum == 0) {
      const std::string kOpcodeStr = packet::toString(static_cast<packet::Opcode>(packet.opcode));
      if (kOpcodeStr.size() >= indentSize) {
        ss << kOpcodeStr.substr(0, indentSize);
      } else {
        ss << kOpcodeStr;
        ss << std::string(indentSize-kOpcodeStr.size(), ' ');
      }
    } else {
      ss << std::string(indentSize, ' ');
    }
    for (int i=startingIndex; ((i < (startingIndex + kBytesPerLine)) && (i < copiedData.size())); ++i) {
      ss << ' ';
      if (isPrintable(copiedData.at(i))) {
        ss << (char)copiedData.at(i);
      } else {
        ss << ' ';
      }
      ss << ' ';
    }
    if (lineNum != lineCount-1) {
      //Indent for next iteration
      ss << '\n';
      ss << std::string(indentSize, ' ');
    }
  }
  ss << '\n';
  std::cout << ss.rdbuf() << std::flush;
}

std::string PacketLogger::getLogFilePath() const {
  return absl::StrFormat("%s\\%d.txt", logFileDirectoryPath_, getMsSinceEpoch(), ".txt");
}