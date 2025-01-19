#ifndef HYPERBOT_HPP_
#define HYPERBOT_HPP_

#include <cstdint>
#include <string_view>

class Hyperbot {
public:
  void connect(std::string_view ipAddress, int32_t port);
};

#endif // HYPERBOT_HPP_
