#ifndef PK2_TEXT_ZONE_NAME_DATA_HPP_
#define PK2_TEXT_ZONE_NAME_DATA_HPP_

#include "../../../common/pk2/ref/textZoneName.hpp"

#include <unordered_map>

namespace pk2 {

class TextZoneNameData {
public:
  void addItem(ref::TextZoneName &&item);
  const std::string& getRegionName(const uint16_t regionId) const;
private:
  std::unordered_map<uint16_t, std::string> regionNames_;
};

} // namespace pk2

#endif // PK2_TEXT_ZONE_NAME_DATA_HPP_