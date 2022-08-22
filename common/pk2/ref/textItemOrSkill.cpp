#include "textItemOrSkill.hpp"

namespace pk2::ref {

std::ostream& operator<<(std::ostream &stream, const TextItemOrSkill &itemOrSkill) {
	stream << "{service:" << (itemOrSkill.service ? "true":"false") << ','
         << "key: \"" << itemOrSkill.key << "\","
         << "english: \"" << itemOrSkill.english << "\"}";
	return stream;
}

} // namespace pk2::ref