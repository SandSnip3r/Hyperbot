/* #include "shared/silkroad_security.h"
#include "shared/stream_utility.h"
#include "packetLogger.hpp"

#include "stdio.h"
#include <iostream>
#include <stdint.h>
#include <fstream>
#include <string>
#include <random>
#include <vector>
#include <chrono>
#include <list>

#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/filesystem.hpp>
#include <boost/function.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/unordered_map.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>

//===
#include <windows.h>
#include <windowsx.h>
#include "../../Common/Common.h"
#include "../../Common/Silkroad.h"
#include <sstream> */

//Handles network events

//Agent server info

//Inject functions
// boost::function<void(uint16_t opcode, StreamUtility & p, bool encrypted)> InjectJoymax;
// boost::function<void(uint16_t opcode, StreamUtility & p, bool encrypted)> InjectSilkroad;
//Bind inject functions (From proxy internal)
// InjectJoymax = boost::bind(&SilkroadConnection::Inject, &serverConnection, _1, _2, _3);
// InjectSilkroad = boost::bind(&SilkroadConnection::Inject, &clientConnection, _1, _2, _3);

#include "configData.hpp"
#include "session.hpp"
#include "iniReader.hpp"
#include "silkroadConnection.hpp"
#include "../../common/divisionInfo.hpp"
#include "../../common/parsing.hpp"
#include "../../common/pk2ReaderModern.hpp"
#include "gameData.hpp"

#include <filesystem>
#include <iostream>

using namespace std;
namespace fs = std::experimental::filesystem::v1;

int main(int argc, char* argv[]) {
	const fs::path kConfigFilePath{"config.ini"};
  try {
    ini::IniReader configReader{kConfigFilePath};
    config::ConfigData configData(configReader);
    pk2::media::GameData gameData(configData.silkroadDirectory());
	  Session session{gameData, configData};
	  session.start();
  } catch (std::exception &ex) {
    cerr << ex.what() << '\n';
    return 2;
  }
	return 0;
}