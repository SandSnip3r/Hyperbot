#include <silkroad_lib/pk2/parsing/helper.hpp>

#include <stdexcept>

namespace sro::pk2::parsing {

StringLineIteratorContainer::StringLineIteratorContainer(const std::string &str, const std::string &delim) : str_(str), delim_(delim) {}

StringLineIteratorContainer::Iterator::Iterator(const StringLineIteratorContainer &container, size_t pos) : container_(container), pos_(pos) {}

StringLineIteratorContainer::Iterator& StringLineIteratorContainer::Iterator::operator++() {
  if (pos_ < container_.str_.size()) {
    size_t nextPos = container_.str_.find(container_.delim_, pos_);
    if (nextPos == std::string::npos) {
      pos_ = container_.str_.size(); // End
    } else {
      pos_ = nextPos + container_.delim_.size(); // Move past delimiter
    }
  }
  return *this;
}

bool StringLineIteratorContainer::Iterator::operator!=(const Iterator &other) const {
  return pos_ != other.pos_;
}

absl::string_view StringLineIteratorContainer::Iterator::operator*() const {
  if (pos_ >= container_.str_.size()) {
    throw std::runtime_error("Iterator out of bounds");
  }
  size_t nextPos = container_.str_.find(container_.delim_, pos_);
  if (nextPos == std::string::npos) {
    nextPos = container_.str_.size();
  }
  return absl::string_view(container_.str_.data() + pos_, nextPos - pos_);
}

StringLineIteratorContainer::Iterator StringLineIteratorContainer::begin() const {
  return Iterator(*this, 0);
}

StringLineIteratorContainer::Iterator StringLineIteratorContainer::end() const {
  return Iterator(*this, str_.size());
}

// std::vector<absl::string_view> splitToStrViews(absl::string_view str, const std::string &delim) {
//   std::vector<absl::string_view> result;
//   size_t start = 0;
//   size_t pos = str.find(delim);
//   while (pos != std::string::npos) {
//     result.emplace_back(str.data() + start, pos - start);
//     start = pos + delim.size();
//     pos = str.find(delim, start);
//   }
//   if (start < str.size()) {
//     result.emplace_back(str.data() + start, str.size() - start);
//   }
//   return result;
// }

} // namespace sro::pk2::parsing