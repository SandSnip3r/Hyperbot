#ifndef PK2_PARSING_HELPER_HPP_
#define PK2_PARSING_HELPER_HPP_

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace sro::pk2::parsing {

class StringLineIteratorContainer {
public:
  StringLineIteratorContainer(const std::string &str, const std::string &delim);

  class Iterator {
  public:
    Iterator(const StringLineIteratorContainer &container, size_t pos);
    Iterator& operator++();
    bool operator!=(const Iterator &other) const;
    std::string_view operator*() const;
  private:
    const StringLineIteratorContainer &container_;
    size_t pos_;
  };

  Iterator begin() const;
  Iterator end() const;
private:
  const std::string &str_;
  const std::string delim_;
};

std::vector<std::string_view> splitToStrViews(std::string_view str, const std::string &delim);

} // namespace sro::pk2::parsing

#endif // PK2_PARSING_HELPER_HPP_