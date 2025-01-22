#include "pk2/parsing/parsing.hpp"
#include "pk2/parsing/regionInfoParser.hpp"

#include <absl/log/log.h>

#include <optional>
#include <stdexcept>
#include <string>

namespace sro::pk2::parsing {

// https://en.wikipedia.org/wiki/Code_page_949_(IBM)
RegionInfo parseRegionInfo(const std::vector<uint8_t> &data) {
  std::string fileAsString(data.begin(), data.end());
  const auto lines = split(fileAsString, "\r\n");

  auto isEmptyLine = [](const std::string &line) {
    // In this file, there are lines which are only a bunch of tabs, ending with CR/LF
    if (line.size() == 0) {
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
  std::optional<RegionInfo::Region> currentRegion;
  auto saveCurrentRegion = [&regionInfo, &currentRegion](){
    regionInfo.regions.emplace_back(*std::move(currentRegion));
  };
  for (const auto &line : lines) {
    if (line[0] == '#') {
      // Start of a section
      if (currentRegion) {
        // We were in a section already, we're done with that now. Add our previous section to the list.
        saveCurrentRegion();
      }
      // Start a new region for this section
      currentRegion.emplace();
      if (line[1] == 'F') {
        // Field
        currentRegion->regionType = RegionInfo::Region::Type::kField;
      } else if (line[1] == 'T') {
        // Town
        currentRegion->regionType = RegionInfo::Region::Type::kTown;
      } else {
        throw std::runtime_error("This is the start of a section, but is an unknown section type");
      }
    } else if (line[0] >= '0' && line[0] <= '9') {
      // Not the start of a section, is a rect definition
      if (!currentRegion) {
        throw std::runtime_error("Found a line which belongs in a section, but we're not in a section");
      }
      // Is another ALL or RECT
      const auto pieces = split(line, "\t");
      if (pieces.size() < 3) {
        throw std::runtime_error("Expecting line to have at least 3 pieces");
      }
      const auto regionX = std::stoi(pieces[0]);
      const auto regionZ = std::stoi(pieces[1]);
      if (pieces[2].size() >= 3 &&
          pieces[2][0] == 'A' &&
          pieces[2][1] == 'L' &&
          pieces[2][2] == 'L') {
        // ALL
        if (pieces.size() != 3) {
          throw std::runtime_error("Expecting ALL line to have exactly 3 pieces");
        }
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
        const auto rectBeginX = std::stoi(pieces[3]);
        const auto rectBeginZ = std::stoi(pieces[4]);
        const auto rectEndX = std::stoi(pieces[5]);
        const auto rectEndZ = std::stoi(pieces[6]);
        currentRegion->regionRects.emplace_back(regionX, regionZ, rectBeginX, rectBeginZ, rectEndX, rectEndZ);
      } else {
        throw std::runtime_error("Unknown rect type");
      }
    } else if (isEmptyLine(line)) {
      // Skip empty lines
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