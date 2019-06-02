#include "../../common/Silkroad.h"

#include <string>

#ifndef LOADER_HPP_
#define LOADER_HPP_

class Loader {
private:
	std::string dllPath;
	const std::string kSilkroadPath;
	SilkroadData sroData;
	std::string arguments;
	std::string clientPath;
public:
	// Construct a Loader with the given silkroad directory
	// The Loader is now ready to start a client
	Loader(const std::string &silkroadPath);
	
	// Starts the Silkroad client process and injects the DLL
	void startClient();

	// TODO
  void killClient();
};

#endif