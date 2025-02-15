#ifndef PK2_MEDIA_ITEM_DATA_HPP_
#define PK2_MEDIA_ITEM_DATA_HPP_

#include <silkroad_lib/pk2/ref/item.hpp>

#include <functional>
#include <unordered_map>

namespace pk2 {

class ItemData {
public:
	using ItemMap = std::unordered_map<sro::pk2::ref::ItemId, sro::pk2::ref::Item>;
	void addItem(sro::pk2::ref::Item &&item);
	bool haveItemWithId(sro::pk2::ref::ItemId id) const;
	const sro::pk2::ref::Item& getItemById(sro::pk2::ref::ItemId id) const;
	const sro::pk2::ref::Item& getItemByCodeName128(const std::string &codeName) const;
  sro::pk2::ref::ItemId getItemId(std::function<bool(const sro::pk2::ref::Item&)> predicate) const;
	const ItemMap::size_type size() const;
private:
	ItemMap items_;
};

} // namespace pk2

#endif // PK2_MEDIA_ITEM_DATA_HPP_