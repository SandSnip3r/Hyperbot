#include <string>

#ifndef LOADER_HPP_
#define LOADER_HPP_

class Loader {
public:
	// Construct a Loader with the given silkroad directory
	// The Loader is now ready to start a client
	Loader(const std::string &silkroadPath, uint8_t locale);
	
	// Starts the Silkroad client process and injects the DLL
	void startClient();

	// TODO
  // void killClient();
private:
	std::string dllPath_;
	const std::string kSilkroadPath_;
  const uint8_t kLocale_;
	std::string arguments_;
	std::string clientPath_;
};

#endif