#ifndef PK2_LEVEL_DATA_HPP_
#define PK2_LEVEL_DATA_HPP_

#include <silkroad_lib/pk2/ref/level.h>

#include <unordered_map>

namespace pk2 {

class LevelData {
public:
	using LevelMap = std::unordered_map<uint8_t, sro::pk2::ref::Level>;
	void addLevelItem(sro::pk2::ref::Level &&level);
	const sro::pk2::ref::Level& getLevel(uint8_t lvl) const;
private:
	LevelMap Levels_;
};

} // namespace pk2

#endif // PK2_LEVEL_DATA_HPP_