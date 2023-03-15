#ifndef PK2_REF_REGION_HPP_
#define PK2_REF_REGION_HPP_

#include "../../../common/pk2/ref/region.hpp"

#include <silkroad_lib/position.h>

#include <unordered_map>

namespace pk2 {

class RefRegion {
public:
	using RegionMap = std::unordered_map<sro::RegionId, ref::Region>;
  void addRegion(ref::Region &&region);
  const RegionMap& regionMap() const;
  const ref::Region& getRegion(sro::RegionId regionId) const;
private:
	RegionMap regions_;
};

} // namespace pk2

#endif // PK2_REF_REGION_HPP_