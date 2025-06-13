#include <silkroad_lib/pk2/parsing/helper.hpp>
#include <silkroad_lib/pk2/parsing/parsing.hpp>
#include <silkroad_lib/pk2/parsing/regionInfoParser.hpp>

#include <absl/log/log.h>
#include <absl/strings/str_format.h>
#include <absl/strings/str_split.h>

#include <charconv>
#include <optional>
#include <stdexcept>
#include <string>

namespace sro::pk2::parsing {

// https://en.wikipedia.org/wiki/Code_page_949_(IBM)
RegionInfo parseRegionInfo(const std::vector<uint8_t> &data) {
  std::string fileAsString(data.begin(), data.end());
  StringLineIteratorContainer lineIteratorContainer(fileAsString, "\r\n");

  auto isEmptyLine = [](absl::string_view line) {
    // In this file, there are lines which are only a bunch of tabs, ending with CR/LF
    if (line.empty()) {
      return true;
    }
    for (char c : line) {
      if (!std::isspace(c)) {
        return false;
      }
    }
    return true;
  };

  RegionInfo regionInfo;
  std::optional<RegionInfo::Continent> currentRegion;
  auto saveCurrentRegion = [&regionInfo, &currentRegion](){
    regionInfo.continents.emplace_back(*std::move(currentRegion));
  };

  for (absl::string_view line : lineIteratorContainer) {
    if (isEmptyLine(line)) {
      // Skip empty lines
      continue;
    }
    if (line[0] == '#') {
      // Start of a section
      if (currentRegion) {
        // We were in a section already, we're done with that now. Add our previous section to the list.
        saveCurrentRegion();
      }
      // Start a new region for this section
      currentRegion.emplace();
      if (line.size() > 1 && line[1] == 'F') {
        // Field
        currentRegion->regionType = RegionInfo::Continent::Type::kField;
      } else if (line.size() > 1 && line[1] == 'T') {
        // Town
        currentRegion->regionType = RegionInfo::Continent::Type::kTown;
      } else {
        throw std::runtime_error("This is the start of a section, but is an unknown section type");
      }
    } else if (line[0] >= '0' && line[0] <= '9') {
      // Not the start of a section, is a rect definition
      if (!currentRegion) {
        throw std::runtime_error("Found a line which belongs in a section, but we're not in a section");
      }
      // Is another ALL or RECT
      const std::vector<absl::string_view> pieces = absl::StrSplit(line, "\t");
      if (pieces.size() < 3) {
        throw std::runtime_error(absl::StrFormat("Expecting line \"%s\" to have at least 3 pieces. It only has %d", line, pieces.size()));
      }
      int regionX;
      int regionZ;
      std::from_chars_result resultRegionX = std::from_chars(pieces[0].data(), pieces[0].data() + pieces[0].size(), regionX);
      std::from_chars_result resultRegionZ = std::from_chars(pieces[1].data(), pieces[1].data() + pieces[1].size(), regionZ);
      if (resultRegionX.ec != std::errc() || resultRegionZ.ec != std::errc()) {
        throw std::runtime_error("Failed to parse region coordinates");
      }
      if (pieces[2].size() >= 3 &&
          pieces[2][0] == 'A' &&
          pieces[2][1] == 'L' &&
          pieces[2][2] == 'L') {
        // ALL
        currentRegion->regionRects.emplace_back(regionX, regionZ);
        {
          const auto &b = currentRegion->regionRects.back();
          if (b.height < 0) {
            LOG(WARNING) << "Negative height. regionX: " << regionX << " ,regionZ: " << regionZ;
          }
        }
      } else if (pieces[2].size() >= 4 &&
                 pieces[2][0] == 'R' &&
                 pieces[2][1] == 'E' &&
                 pieces[2][2] == 'C' &&
                 pieces[2][3] == 'T') {
        // RECT
        if (pieces.size() != 7) {
          throw std::runtime_error("Expecting RECT line to have exactly 7 pieces");
        }
        int rectBeginX;
        int rectBeginZ;
        int rectEndX;
        int rectEndZ;
        std::from_chars_result resultRectBeginX = std::from_chars(pieces[3].data(), pieces[3].data() + pieces[3].size(), rectBeginX);
        std::from_chars_result resultRectBeginZ = std::from_chars(pieces[4].data(), pieces[4].data() + pieces[4].size(), rectBeginZ);
        std::from_chars_result resultRectEndX = std::from_chars(pieces[5].data(), pieces[5].data() + pieces[5].size(), rectEndX);
        std::from_chars_result resultRectEndZ = std::from_chars(pieces[6].data(), pieces[6].data() + pieces[6].size(), rectEndZ);
        if (resultRectBeginX.ec != std::errc() ||
            resultRectBeginZ.ec != std::errc() ||
            resultRectEndX.ec != std::errc() ||
            resultRectEndZ.ec != std::errc()) {
          throw std::runtime_error("Failed to parse rect coordinates");
        }
        currentRegion->regionRects.emplace_back(regionX, regionZ, rectBeginX, rectBeginZ, rectEndX, rectEndZ);
      } else {
        throw std::runtime_error("Unknown rect type");
      }
    } else {
      throw std::runtime_error("Unknown data on line");
    }
  }

  if (currentRegion) {
    // We have one last section to save
    saveCurrentRegion();
  }

  return regionInfo;
}

} // namespace sro::pk2::parsing