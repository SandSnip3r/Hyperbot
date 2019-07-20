#ifndef CONFIG_CHARACTER_LOGIN_DATA_HPP_
#define CONFIG_CHARACTER_LOGIN_DATA_HPP_

#include <string>

namespace config {

struct CharacterLoginData {
public:
  std::string name;
  std::string id;
  std::string password;
};

} // namespace config

#endif // CONFIG_CHARACTER_LOGIN_DATA_HPP_