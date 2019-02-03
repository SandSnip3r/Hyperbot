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
	Loader(const std::string &silkroadPath);
	void startClient();
  void killClient();
};

#endif