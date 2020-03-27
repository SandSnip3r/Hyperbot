#ifndef PK2_MEDIA_ITEM_HPP_
#define PK2_MEDIA_ITEM_HPP_

#include <string>
#include <ostream>

namespace pk2::media {

using ItemId = uint32_t;

struct Item {
	ItemId id;
	std::string codeName128;
	uint8_t cashItem;
	uint8_t bionic;
	uint8_t typeId1;
	uint8_t typeId2;
	uint8_t typeId3;
	uint8_t typeId4;
	int32_t maxStack;
	int32_t param1;
	int32_t param2;
	int32_t param4;
};

std::ostream& operator<<(std::ostream &stream, const Item &item);

} // namespace pk2::media

#endif // PK2_MEDIA_ITEM_HPP_