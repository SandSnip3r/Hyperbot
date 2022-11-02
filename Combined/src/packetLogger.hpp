#ifndef PACKETLOGGER_H_
#define PACKETLOGGER_H_

#include "shared/silkroad_security.h"
#include "shared/stream_utility.h"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

class PacketLogger {
private:
	const std::string directoryPath;
	std::string filePath;
	std::ofstream logfile;
	bool logToFile{true};
	bool logToConsole{true};
	int64_t getMsSinceEpoch() const;
	bool isPrintable(uint8_t data) const;
	void logPacketToFile(int64_t msSinceEpoch, const PacketContainer &packet, bool blocked, PacketContainer::Direction direction);
	void logPacketToConsole(int64_t msSinceEpoch, const PacketContainer &packet, bool blocked, PacketContainer::Direction direction);
public:
	PacketLogger(const std::string &logDirectoryPath);
	void logPacket(const PacketContainer &packet, bool blocked, PacketContainer::Direction direction);
};

#endif