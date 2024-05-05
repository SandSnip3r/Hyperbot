#ifndef STATE_MACHINE_LOGIN_HPP_
#define STATE_MACHINE_LOGIN_HPP_

// #include "broker/eventBroker.hpp"
#include "stateMachine.hpp"

// #include "type_id/typeCategory.hpp"

// #include <silkroad_lib/scalar_types.h>

// #include <optional>
#include <array>
#include <cstdint>
#include <string>

namespace state::machine {

class Login : public StateMachine {
public:
  Login(Bot &bot, const std::string &username, const std::string &password, const std::string &characterName);
  ~Login() override;
  void onUpdate(const event::Event *event) override;
  bool done() const override;
private:
  // TODO: The two things below do not belong here
  static inline const std::array<uint8_t,6> kMacAddress = {0,0,0,0,0,0};
  static inline const std::string kCaptchaAnswer = "";

  static inline std::string kName{"Login"};
  const std::string username_;
  const std::string password_;
  const std::string characterName_;
  bool done_{false};
};

} // namespace state::machine

#endif // STATE_MACHINE_LOGIN_HPP_