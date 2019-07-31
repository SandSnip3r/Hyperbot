#ifndef PK2_MEDIA_ITEM_DATA_HPP_
#define PK2_MEDIA_ITEM_DATA_HPP_

#include "../../common/item.hpp"

#include <unordered_map>

namespace pk2::media {

class ItemData {
public:
	using ItemMap = std::unordered_map<ItemId,Item>;
	void addItem(Item &&item);
	bool haveItemWithId(ItemId id) const;
	const Item& getItemById(ItemId id) const;
	const ItemMap::size_type size() const;
private:
	ItemMap items_;
};

} // namespace pk2::media

#endif // PK2_MEDIA_ITEM_DATA_HPP_