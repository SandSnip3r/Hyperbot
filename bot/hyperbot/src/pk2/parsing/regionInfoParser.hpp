#ifndef PK2_PARSING_REGION_INFO_PARSER_HPP_
#define PK2_PARSING_REGION_INFO_PARSER_HPP_

#include "pk2/regionInfo.hpp"

#include <cstdint>
#include <vector>

namespace pk2::parsing {

RegionInfo parseRegionInfo(const std::vector<uint8_t> &data);

} // namespace pk2::parsing

#endif // PK2_PARSING_REGION_INFO_PARSER_HPP_