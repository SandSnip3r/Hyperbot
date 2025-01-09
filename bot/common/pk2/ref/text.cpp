#include "text.hpp"

namespace pk2::ref {

std::ostream& operator<<(std::ostream &stream, const Text &text) {
	stream << "{service:" << (text.service ? "true":"false") << ','
         << "key: \"" << text.key << "\","
         << "english: \"" << text.english << "\"}";
	return stream;
}

} // namespace pk2::ref