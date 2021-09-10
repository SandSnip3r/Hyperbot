#ifndef PK2_READER_MODERN_HPP
#define PK2_READER_MODERN_HPP

#include "pk2.h"
#include "pk2Reader.h"

#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

namespace pk2 {

namespace fs = std::filesystem;

class Pk2ReaderModern {
public:
  Pk2ReaderModern(const fs::path &pk2Path);
  ~Pk2ReaderModern();
  PK2Entry getEntry(const std::string &entryName);
  std::vector<uint8_t> getEntryData(PK2Entry &entry);
  std::vector<char> getEntryDataChar(PK2Entry &entry);
private:
  std::mutex mutex_;
  fs::path pk2Path_;
  PK2Reader pk2Reader_;
};

} // namespace pk2

#endif // PK2_READER_MODERN_HPP