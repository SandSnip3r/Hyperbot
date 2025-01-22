#include "pk2/pk2ReaderModern.hpp"

#include <absl/log/log.h>
#include <absl/strings/str_format.h>

#include <filesystem>
#include <stdexcept>

namespace sro::pk2 {

Pk2ReaderModern::Pk2ReaderModern(const fs::path &pk2Path) : pk2Path_(pk2Path) {
  if (!fs::exists(pk2Path_)) {
    throw std::runtime_error(absl::StrFormat("PK2 file \"%s\" does not exist", pk2Path_.string()));
  }
  bool pk2OpenResult = pk2Reader_.Open(pk2Path_.string().c_str());
  if (!pk2OpenResult) {
    throw std::runtime_error(absl::StrFormat("PK2Reader failed to open pk2 file \"%s\" (PK2Reader: \"%s\")", pk2Path_.string(), pk2Reader_.GetError()));
  }
}

Pk2ReaderModern::~Pk2ReaderModern() {
  pk2Reader_.Close();
}

bool Pk2ReaderModern::hasEntry(const std::string &entryName) {
  std::unique_lock<std::mutex> lock(mutex_);
  PK2Entry entry = {0};
  return pk2Reader_.GetEntry(entryName.c_str(), entry);
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

std::vector<uint8_t> Pk2ReaderModern::getEntryData(PK2Entry &entry) { // TODO: This function should take a reference to a vector as input instead
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

void Pk2ReaderModern::clearCache() {
  std::unique_lock<std::mutex> lock(mutex_);
  pk2Reader_.ClearCache();
}


} // namespace sro::pk2