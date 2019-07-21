/* Shout out to Weeman! */

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

void generateBlankConfig(const fs::path &path);

int main(int argc, char* argv[]) {
	const fs::path kConfigFilePath{"config.ini"};
  if (!fs::exists(kConfigFilePath)) {
    cerr << "Cannot find config file at path \"" << kConfigFilePath << "\". Generating template at this path. Please open and edit it.\n";
    generateBlankConfig(kConfigFilePath);
    return 1;
  }
  try {
    ini::IniReader configReader{kConfigFilePath};
    config::ConfigData configData(configReader);
    pk2::media::GameData gameData(configData.silkroadDirectory());
	  Session session{gameData, configData.silkroadDirectory(), configData.characterLoginData()};
	  session.start();
  } catch (std::exception &ex) {
    cerr << ex.what() << '\n';
    return 2;
  }
	return 0;
}

void generateBlankConfig(const fs::path &path) {
  ofstream outFile(path);
  if (!outFile) {
    cerr << "Unable to open new file at path \"" << path << "\"\n";
  }
  outFile << "; This is the Hyperbot config file\n";
  outFile << "\n";
  outFile << "; Provide the path to the silkroad directory\n";
  outFile << "[Silkroad]\n";
  outFile << "path=\n";
  outFile << "\n";
  outFile << "; One section is required where the section name is the character name\n";
  outFile << "; Replace \"CharacterName\" with your character's name\n";
  outFile << "[CharacterName]\n";
  outFile << "id=\n";
  outFile << "password=\n";
}