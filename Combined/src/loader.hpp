#ifndef LOADER_HPP_
#define LOADER_HPP_

#include "../../common/divisionInfo.hpp"

#include <filesystem>
#include <string>

namespace {
namespace fs = std::experimental::filesystem::v1;
}

class Loader {
public:
	// Construct a Loader with the given silkroad directory
	// The Loader is now ready to start a client
	Loader(const fs::path &kSilkroadDirectoryPath, const pk2::DivisionInfo &divisionInfo);
	
	// Starts the Silkroad client process and injects the DLL
	void startClient();

	// TODO
  // void killClient();
private:
	const fs::path kSilkroadDirectoryPath_;
  const pk2::DivisionInfo &kDivisionInfo_;
	std::string dllPath_;
	std::string arguments_;
	std::string clientPath_;
};

#endif