#ifndef STATE_MACHINE_LOGIN_HPP_
#define STATE_MACHINE_LOGIN_HPP_

#include "characterLoginInfo.hpp"
#include "state/machine/stateMachine.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

namespace state::machine {

class Login : public StateMachine {
public:
  Login(Bot &bot, const CharacterLoginInfo &characterLoginInfo);
  Login(StateMachine *parent, const CharacterLoginInfo &characterLoginInfo);
  ~Login() override;
  Status onUpdate(const event::Event *event) override;
private:
  // TODO: The two things below do not belong here
  static inline const std::array<uint8_t,6> kMacAddress = {0,0,0,0,0,0};
  static inline const std::string kCaptchaAnswer = "";

  static inline std::string kName{"Login"};
  const std::string username_;
  const std::string password_;
  const std::string characterName_;
  bool initialized_{false};
  std::optional<uint32_t> agentServerToken_;
  bool waitingOnShardList_{false};
  bool loginRequestSent_{false};
  bool spawned_{false};
};

} // namespace state::machine

#endif // STATE_MACHINE_LOGIN_HPP_