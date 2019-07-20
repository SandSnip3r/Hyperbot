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

#include "session.hpp"
#include "silkroadConnection.hpp"
#include "../../common/divisionInfo.hpp"
#include "../../common/parsing.hpp"
#include "../../common/pk2ReaderModern.hpp"
#include "gameData.hpp"

#include <iostream>

using namespace std;

//Packet from server when char already logged in
//41218 02 03
int main(int argc, char* argv[]) {
	//Let the user know which ports to connect to
	cout << "Redirect Silkroad to 127.0.0.1:" << Config::BindPort << endl;
	cout << "Redirect the bot to 127.0.0.1:" << Config::BotBind << endl << endl;

	const string kConfigFilePath{"config.ini"};
  // Parse config into an object

  // Use config to get silkroad path
  // Find and parse the Media.pk2
  // Parse the Media.pk2 into some game data
	const std::string kSilkroadDirectory = "C:\\Program Files (x86)\\Evolin\\"; // TODO: Get from config file

  try {
    pk2::media::GameData gameData(kSilkroadDirectory);

    // Pass game data object to the Session
    // Pass config object to the Session
	  Session session{gameData, kSilkroadDirectory};

	  session.start();
  } catch (std::exception &ex) {
    cerr << ex.what() << '\n';
    return 1;
  }
	return 0;
}