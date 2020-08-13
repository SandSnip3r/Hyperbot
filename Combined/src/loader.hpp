#ifndef LOADER_HPP_
#define LOADER_HPP_

#include "../../common/pk2/divisionInfo.hpp"

#include <filesystem>
#include <string>

class Loader {
public:
	// Construct a Loader with the given silkroad directory
	// The Loader is now ready to start a client
	Loader(const std::experimental::filesystem::v1::path &kSilkroadDirectoryPath, const pk2::DivisionInfo &divisionInfo);
	
	// Starts the Silkroad client process and injects the DLL
	void startClient(uint16_t proxyListeningPort);

	// TODO
  // void killClient();
private:
	const std::experimental::filesystem::v1::path kSilkroadDirectoryPath_;
  const pk2::DivisionInfo &kDivisionInfo_;
	std::string dllPath_;
	std::string arguments_;
	std::string clientPath_;
};

#endif