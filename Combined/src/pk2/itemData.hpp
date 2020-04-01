#ifndef PK2_MEDIA_ITEM_DATA_HPP_
#define PK2_MEDIA_ITEM_DATA_HPP_

#include "../../../common/pk2/ref/item.hpp"

#include <unordered_map>

namespace pk2 {

class ItemData {
public:
	using ItemMap = std::unordered_map<ref::ItemId,ref::Item>;
	void addItem(ref::Item &&item);
	bool haveItemWithId(ref::ItemId id) const;
	const ref::Item& getItemById(ref::ItemId id) const;
	const ItemMap::size_type size() const;
private:
	ItemMap items_;
};

} // namespace pk2

#endif // PK2_MEDIA_ITEM_DATA_HPP_