#include "iniReader.hpp"

#include <fstream>
#include <regex>

namespace ini {

bool IniReader::valueExists(const std::string &kSectionName, const std::string &kKey) const {
  auto sectionMapIt = sectionMap_.find(kSectionName);
  if (sectionMapIt == sectionMap_.end()) {
    return false;
  }
  return (sectionMapIt->second.find(kKey) != sectionMapIt->second.end());
}

std::vector<std::string> IniReader::getSections() const {
  std::vector<std::string> sections;
  for (const auto &i : sectionMap_) {
    sections.emplace_back(i.first);
  }
  return sections;
}

const std::string& IniReader::getStr(const std::string &kSectionName, const std::string &kKey) const {
  return sectionMap_.at(kSectionName).at(kKey);
}

IniReader::IniReader(const std::experimental::filesystem::v1::path &kPath) {
  parseConfig(kPath);
}

namespace {
  std::string removeCarriageReturn(std::string str) {
    str.erase(std::remove(str.begin(), str.end(), '\r'), str.end());
    return str;
  }
}

void IniReader::parseConfig(const std::experimental::filesystem::v1::path &kPath) {
  std::ifstream inFile(kPath);
  if (!inFile) {
    throw std::runtime_error("Unable to locate config file \""+kPath.string()+"\"");
  }

  std::regex iniSectionRegex(R"delim(\[(.*)\]\s*)delim");
  std::regex iniCommentRegex(R"delim(;.*\s*)delim");
  std::regex iniKeyValueRegex(R"delim(([^\s]+)=(.*?)\s*\r?)delim");
  std::regex iniWhitespaceRegex(R"delim(\s*)delim");

  std::string currentSection;
  KeyValueMap *keyValueMap{nullptr};
  std::string line;
  // Read lines one-by-one
  int lineNum=0;
  while (getline(inFile, line)) {
    ++lineNum;
    std::smatch matchResult;

    if (std::regex_match(line, matchResult, iniSectionRegex)) {
      // Found a section
      const std::string &section = matchResult[1].str();
      if (sectionMap_.find(section) == sectionMap_.end()) {
        // First occurence of this section, insert it into our map
         auto itBoolResult = sectionMap_.emplace(section, KeyValueMap());
         if (!itBoolResult.second) {
           throw std::runtime_error("Failed to create section \""+section+"\"");
         }
         // Update the current section and pointer to the key=value map
         currentSection = section;
         keyValueMap = &itBoolResult.first->second;
      } else {
        throw std::runtime_error(kPath.string()+":"+std::to_string(lineNum)+" Section with name \""+currentSection+"\" occurs multiple times");
      }
    } else if (std::regex_match(line, matchResult, iniKeyValueRegex)) {
      // Key=value entry
      if (currentSection == "" || keyValueMap == nullptr) {
        throw std::runtime_error(kPath.string()+":"+std::to_string(lineNum)+" Key=value pair is not allowed outside of a section");
      }
      const std::string &key = matchResult[1].str();
      const std::string &value = matchResult[2].str();
      if (keyValueMap->find(key) == keyValueMap->end()) {
        // Insert key/value into map
        keyValueMap->emplace(key,value);
      } else {
        throw std::runtime_error(kPath.string()+":"+std::to_string(lineNum)+" Value \""+keyValueMap->at(key)+"\" already exists for key \""+key+"\"");
      }
    } else if (std::regex_match(line, matchResult, iniWhitespaceRegex)) {
      // Ignore whitespace-only lines
    } else if (std::regex_match(line, matchResult, iniCommentRegex)) {
      // Ignore comment lines
    } else {
      // Unknown
      throw std::runtime_error(kPath.string()+":"+std::to_string(lineNum)+" Line \""+removeCarriageReturn(line)+"\" is not a section, key=value pair, comment line, or blank lane.");
    }
  }
}

} // namespace ini