#ifndef LOADER_HPP_
#define LOADER_HPP_

#include "../../common/pk2/divisionInfo.hpp"

#include <filesystem>
#include <string>
#include <string_view>

class Loader {
public:
	// Construct a Loader with the given silkroad directory
	// The Loader is now ready to start a client
	Loader(std::string_view clientPath, const pk2::DivisionInfo &divisionInfo);
	
	// Starts the Silkroad client process and injects the DLL
	void startClient(uint16_t proxyListeningPort);

	// TODO
  // void killClient();
private:
  const pk2::DivisionInfo &kDivisionInfo_;
	std::filesystem::path dllPath_;
	std::filesystem::path clientPath_;
	std::string arguments_;
};

#endif