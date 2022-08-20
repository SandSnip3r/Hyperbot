#ifndef PK2_LEVEL_DATA_HPP_
#define PK2_LEVEL_DATA_HPP_

#include "../../../common/pk2/ref/level.hpp"

#include <unordered_map>

namespace pk2 {

class LevelData {
public:
	using LevelMap = std::unordered_map<uint8_t, ref::Level>;
	void addLevelItem(ref::Level &&level);
	const ref::Level& getLevel(uint8_t lvl) const;
private:
	LevelMap Levels_;
};

} // namespace pk2

#endif // PK2_LEVEL_DATA_HPP_