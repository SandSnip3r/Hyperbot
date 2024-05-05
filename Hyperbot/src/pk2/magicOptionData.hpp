#ifndef PK2_MAGIC_OPTION_DATA_HPP_
#define PK2_MAGIC_OPTION_DATA_HPP_

#include "../../../common/pk2/ref/magicOption.hpp"

#include <unordered_map>

namespace pk2 {

class MagicOptionData {
public:
	using MagicOptionMap = std::unordered_map<ref::MagicOptionId, ref::MagicOption>;
	void addItem(ref::MagicOption &&magOpt);
	const ref::MagicOption& getMagicOptionById(ref::MagicOptionId id) const;
private:
	MagicOptionMap magicOptions_;
};

} // namespace pk2

#endif // PK2_MAGIC_OPTION_DATA_HPP_