#ifndef SRO_PK2_READER_MODERN_H_
#define SRO_PK2_READER_MODERN_H_

#include "pk2.h"
#include "pk2Reader.h"

#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

namespace sro::pk2 {

namespace fs = std::filesystem;

class Pk2ReaderModern {
public:
  Pk2ReaderModern(const fs::path &pk2Path);
  ~Pk2ReaderModern();
  bool hasEntry(const std::string &entryName);
  PK2Entry getEntry(const std::string &entryName);
  std::vector<uint8_t> getEntryData(PK2Entry &entry);
  std::vector<char> getEntryDataChar(PK2Entry &entry);
  void clearCache();
private:
  std::mutex mutex_;
  fs::path pk2Path_;
  PK2Reader pk2Reader_;
};

} // namespace sro::pk2

#endif // SRO_PK2_READER_MODERN_H_