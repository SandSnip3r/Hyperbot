#ifndef PACKET_LOGGER_HPP_
#define PACKET_LOGGER_HPP_

#include "shared/silkroad_security.h"
#include "shared/stream_utility.h"

#include <fstream>
#include <string>

class PacketLogger {
public:
	PacketLogger(const std::string &logDirectoryPath);
	void logPacket(const PacketContainer &packet, bool blocked, PacketContainer::Direction direction);
private:
	const std::string directoryPath;
	std::string filePath;
	std::ofstream logfile;
	bool logToFile{true};
	bool logToConsole{false};
	void logPacketToFile(int64_t msSinceEpoch, const PacketContainer &packet, bool blocked, PacketContainer::Direction direction);
	void logPacketToConsole(int64_t msSinceEpoch, const PacketContainer &packet, bool blocked, PacketContainer::Direction direction);
};

#endif // PACKET_LOGGER_HPP_