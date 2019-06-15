#include "opcode.hpp"
#include "packetLogger.hpp"

int64_t PacketLogger::getMsSinceEpoch() const {
  std::chrono::time_point<std::chrono::system_clock> nowTimePoint = std::chrono::system_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(nowTimePoint.time_since_epoch()).count();
}

bool PacketLogger::isPrintable(uint8_t data) const {
  return (data >= ' ' && data <= '~');
}

void PacketLogger::logPacketToFile(int64_t msSinceEpoch, const PacketContainer &packet, bool blocked, Direction direction) {
  if (!logfile) {
    throw std::runtime_error("Log file \""+filePath+"\" problem");
  }
  std::stringstream ss;
  ss << msSinceEpoch << ',';
  ss << blocked << ',';
  ss << (int)packet.encrypted << ',';
  ss << (int)packet.massive << ',';
  if (direction == Direction::kClientToServer) {
    ss << "C,";
  } else if (direction == Direction::kServerToClient) {
    ss << "S,";
  } else if (direction == Direction::BotToServer) {
    ss << "B,";
  } else if (direction == Direction::BotToClient) {
    ss << "b,";
  }
  ss << (int)packet.opcode;
  StreamUtility stream = packet.data;
  int byteCount = stream.GetStreamSize();
  for (int i=0; i<byteCount; ++i) {
    ss << ',' << (int)stream.Read<uint8_t>();
  }
  ss << '\n';
  logfile << ss.str() << std::flush;
}

void PacketLogger::logPacketToConsole(int64_t msSinceEpoch, const PacketContainer &packet, bool blocked, Direction direction) {
  const int kBytesPerLine{20};

  std::stringstream ss;
  ss << '[' << msSinceEpoch << "] ";
  if (direction == Direction::kClientToServer) {
    ss << " (C->S)";
  } else if (direction == Direction::kServerToClient) {
    ss << " (S->C)";
  } else if (direction == Direction::BotToServer) {
    ss << " (B->S)";
  } else if (direction == Direction::BotToClient) {
    ss << " (B->C)";
  }
  ss << (int)blocked << ',';
  ss << (int)packet.encrypted << ',';
  ss << (int)packet.massive << ',';
  ss << std::hex << packet.opcode << ' ';
  const int indentSize = ss.str().size();
  StreamUtility stream = packet.data;
  const auto &dataVector = stream.GetStreamVector();
  std::vector<uint8_t> copiedData(dataVector.begin(), dataVector.end());
  const int lineCount = std::ceil((float)copiedData.size()/kBytesPerLine);
  for (int lineNum=0; lineNum<lineCount; ++lineNum) {
    const int startingIndex = kBytesPerLine*lineNum;
    for (int i=startingIndex; ((i < (startingIndex + kBytesPerLine)) && (i < copiedData.size())); ++i) {
      ss << std::setfill('0') << std::setw(2) << std::hex << (int)copiedData.at(i) << ' ';
    }
    ss << '\n';
    if (lineNum == 0) {
      const std::string kOpcodeStr = OpcodeHelp::toStr(static_cast<Opcode>(packet.opcode));
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

PacketLogger::PacketLogger(const std::string &logDirectoryPath) : directoryPath(logDirectoryPath) {
  //TODO: Proper path handling. std::fs? boost?
  filePath = directoryPath + "\\" + std::to_string(getMsSinceEpoch()) + ".txt";
  logfile.open(filePath);
  if (!logfile) {
    throw std::runtime_error("Unable to initialize logfile \""+filePath+"\"");
  }
}

void PacketLogger::logPacket(const PacketContainer &packet, bool blocked, Direction direction) {
  int64_t msSinceEpoch = getMsSinceEpoch();
  if (logToFile) {
    logPacketToFile(msSinceEpoch, packet, blocked, direction);
  }
  if (logToConsole) {
    logPacketToConsole(msSinceEpoch, packet, blocked, direction);
  }
}