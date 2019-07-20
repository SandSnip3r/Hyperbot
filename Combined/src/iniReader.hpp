#ifndef INI_READER_HPP__
#define INI_READER_HPP__

#include <filesystem>
#include <string>
#include <sstream>
#include <unordered_map>
#include <type_traits>
#include <vector>

namespace ini {

class IniReader {
public:
  // Constructed with path to config file.
  // Will parse the config file into memory
  IniReader(const std::experimental::filesystem::v1::path &kPath);

  // Get all section titles
  std::vector<std::string> getSections() const;

  // Query if a key exists in a section
  bool valueExists(const std::string &kSectionName, const std::string &kKey) const;

  // Get a key in a specific section as a specified type
  template<typename T, typename = std::enable_if_t<std::is_same_v<T, std::string>>>
  T get(const std::string &kSectionName, const std::string &kKey) const {
    std::stringstream ss{getStr(kSectionName, kKey)};
    T data;
    ss >> data;
    return data;
  }

  template<>
  std::string get<std::string>(const std::string &kSectionName, const std::string &kKey) const {
    return getStr(kSectionName, kKey);
  }

private:
  using KeyValueMap = std::unordered_map<std::string, std::string>;
  using SectionDataMap = std::unordered_map<std::string, KeyValueMap>;
  SectionDataMap sectionMap_;
  
  // Reads data into the SectionDataMap data structure
  void parseConfig(const std::experimental::filesystem::v1::path &kPath);
  const std::string& getStr(const std::string &kSectionName, const std::string &kKey) const;
};

} // namespace ini

#endif // INI_READER_HPP__