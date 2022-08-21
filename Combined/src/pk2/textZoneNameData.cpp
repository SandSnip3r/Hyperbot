#include "textZoneNameData.hpp"

#include "logging.hpp"

#include <cctype>
#include <string_view>

namespace pk2 {

namespace {

bool isInt(std::string_view str) {
  for (const char c : str) {
    if (!isdigit(c)) {
      return false;
    }
  }
  return true;
}

} // anonymous namespace

void TextZoneNameData::addItem(ref::TextZoneName &&item) {
  if (item.service != 0 && isInt(item.key)) {
    uint16_t regionId = stoi(item.key);
    regionNames_.emplace(regionId, item.english);
  }
}

const std::string& TextZoneNameData::getRegionName(const uint16_t regionId) const {
  const auto it = regionNames_.find(regionId);
  if (it == regionNames_.end()) {
    throw std::runtime_error("Could not find name for region");
  }
  return it->second;
}

} // namespace pk2