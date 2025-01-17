#ifndef SRO_PK2_PARSING_REGION_INFO_PARSER_HPP_
#define SRO_PK2_PARSING_REGION_INFO_PARSER_HPP_

#include "pk2/regionInfo.h"

#include <cstdint>
#include <vector>

namespace sro::pk2::parsing {

RegionInfo parseRegionInfo(const std::vector<uint8_t> &data);

} // namespace sro::pk2::parsing

#endif // SRO_PK2_PARSING_REGION_INFO_PARSER_HPP_