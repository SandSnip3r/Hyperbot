#ifndef PK2_MAGIC_OPTION_DATA_HPP_
#define PK2_MAGIC_OPTION_DATA_HPP_

#include <silkroad_lib/pk2/ref/magicOption.hpp>

#include <unordered_map>

namespace pk2 {

class MagicOptionData {
public:
	using MagicOptionMap = std::unordered_map<sro::pk2::ref::MagicOptionId, sro::pk2::ref::MagicOption>;
	void addItem(sro::pk2::ref::MagicOption &&magOpt);
	const sro::pk2::ref::MagicOption& getMagicOptionById(sro::pk2::ref::MagicOptionId id) const;
private:
	MagicOptionMap magicOptions_;
};

} // namespace pk2

#endif // PK2_MAGIC_OPTION_DATA_HPP_