#ifndef SRO_PK2_PARSING_REGION_INFO_PARSER_HPP_
#define SRO_PK2_PARSING_REGION_INFO_PARSER_HPP_

#include <silkroad_lib/pk2/regionInfo.hpp>

#include <cstdint>
#include <vector>

namespace sro::pk2::parsing {

RegionInfo parseRegionInfo(const std::vector<uint8_t> &data);

} // namespace sro::pk2::parsing

#endif // SRO_PK2_PARSING_REGION_INFO_PARSER_HPP_