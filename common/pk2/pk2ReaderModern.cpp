#include "pk2ReaderModern.hpp"

#include <filesystem>
#include <stdexcept>

namespace pk2 {

Pk2ReaderModern::Pk2ReaderModern(const fs::path &pk2Path) : pk2Path_(pk2Path) {
  if (!fs::exists(pk2Path_)) {
    throw std::runtime_error("PK2 file \""+pk2Path_.string()+"\" does not exist");
  }
  bool pk2OpenResult = pk2Reader_.Open(pk2Path_.string().c_str());
  if (!pk2OpenResult) {
    throw std::runtime_error("PK2Reader failed to open pk2 file \""+pk2Path_.string()+"\" (PK2Reader: \""+pk2Reader_.GetError()+"\")");
  }
}

Pk2ReaderModern::~Pk2ReaderModern() {
  pk2Reader_.Close();
}

PK2Entry Pk2ReaderModern::getEntry(const std::string &entryName) {
  std::unique_lock<std::mutex> lock(mutex_);
  PK2Entry entry = {0};
  bool result = pk2Reader_.GetEntry(entryName.c_str(), entry);
  if (!result) {
    throw std::runtime_error("PK2Reader failed to get entry \""+entryName+"\" (PK2Reader: \""+pk2Reader_.GetError()+"\")");
  }
  return entry;
}

std::vector<uint8_t> Pk2ReaderModern::getEntryData(PK2Entry &entry) {
  std::unique_lock<std::mutex> lock(mutex_);
  std::vector<uint8_t> data;
  bool result = pk2Reader_.ExtractToMemory(entry, data);
  if (!result) {
    throw std::runtime_error("PK2Reader failed to extract data from entry \""+std::string(entry.name)+"\" (PK2Reader: \""+pk2Reader_.GetError()+"\")");
  }
  return data; 
}

std::vector<char> Pk2ReaderModern::getEntryDataChar(PK2Entry &entry) {
  std::unique_lock<std::mutex> lock(mutex_);
  std::vector<char> data;
  bool result = pk2Reader_.ExtractToMemoryChar(entry, data);
  if (!result) {
    throw std::runtime_error("PK2Reader failed to extract data from entry \""+std::string(entry.name)+"\" (PK2Reader: \""+pk2Reader_.GetError()+"\")");
  }
  return data;
}

} // namespace pk2