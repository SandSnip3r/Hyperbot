#include "textZoneNameData.hpp"

#include <absl/log/log.h>
#include <absl/strings/str_format.h>

#include <charconv>
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

void TextZoneNameData::addItem(sro::pk2::ref::TextZoneName &&item) {
  if (item.service != 0 && isInt(item.codeName128)) {
    uint16_t regionId;
    auto [ptr, ec] = std::from_chars(item.codeName128.data(), item.codeName128.data() + item.codeName128.size(), regionId);
    if (ec != std::errc{}) {
      VLOG(1) << absl::StrFormat("Given a text zone name which does not look like we expect. CodeName128: \"%s\"", item.codeName128);
      return;
    }
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