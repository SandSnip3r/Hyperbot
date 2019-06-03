#include <iostream>
#include <functional>
#include <ostream>

using namespace std;

class Packet {
public:
  enum class Opcode { kServerListing, kClientCafe };
  enum class Direction { kClientToServer, kServerToClient };
private:
  Opcode opcode_;
public:
  Opcode opcode() const { return opcode_; }
  Packet(Opcode op) : opcode_(op) {}
};

class LoginServerListPacket {
private:
  uint16_t shardId_;
public:
  LoginServerListPacket(const Packet &packet) {
    // Parse packet
    shardId_ = 123;
  }
  uint16_t shardId() const { return shardId_; }
};

class LoginClientAuthPacket {
private:
  uint8_t locale_;
  // uint16_t kUsername_.size()
  std::string username_;
  // uint16_t kPassword_.size()
  std::string password_;
  uint16_t shardId_;
public:
  LoginClientAuthPacket(uint8_t locale, std::string username, std::string password, uint16_t shardId) : locale_(locale), username_(username), password_(password), shardId_(shardId) {}
  friend std::ostream& operator<<(std::ostream &stream, const LoginClientAuthPacket &packet);
};

std::ostream& operator<<(std::ostream &stream, const LoginClientAuthPacket &packet) {
  stream << "{ locale:" << packet.locale_ << ", username:\"" << packet.username_ << "\", password:\"" << packet.password_ << "\", shardId:" << packet.shardId_ << " }";
  return stream;
}

class LoginModule {
  enum class State {
    kInit,
    kReceivedServerListing,
    kSentLoginAuth/*,...*/
  };
private:
  const uint8_t kLocale{0x16};
  const std::string kUsername{"Example"};
  const std::string kPassword{"asdf"};
  State state_{State::kInit};
  uint16_t shardId_;
  bool autoLogin_{false};
private:
  /// Server sent a list of all gameservers.
  /// Extract {
  ///   uint16_t shardId
  /// }
  /// TODO: Use return types for error conditions
  void serverListReceived(const Packet &packet) {
    //Parse packet by constructing specific packet type
    LoginServerListPacket serverList(packet);
    shardId_ = serverList.shardId();
  }
  /// Login character
  /// TODO: Use return types for error conditions
  void login() {
    //Create new packet with info from parsed packet
    LoginClientAuthPacket clientAuth( kLocale, kUsername, kPassword, shardId_ );
    //Inject packet into stream
    // TODO: Handle injection, upcasting
    std::cout << "Injecting " << clientAuth << '\n';
    // inject(clientAuth, Packet::Direction::kClientToServer);
  }
public:
  void handlePacket(const Packet &packet) {
    if (state_ == State::kInit) {
      if (packet.opcode() == Packet::Opcode::kServerListing) {
        serverListReceived(packet);
        std::cout << "Received serverlist\n";
        state_ = State::kReceivedServerListing;
      }
    } else if (state_ == State::kReceivedServerListing) {
      if (packet.opcode() == Packet::Opcode::kClientCafe) {
        //Dont care about anything in the "cafe" packet. Dont parse it
        if (autoLogin_) {
          login();
          state_ = State::kSentLoginAuth;
        } else {
          std::cout << "0xCAFE received. Dont care, not logging in.\n";
        }
      }
    } else if (state_ == State::kSentLoginAuth) {
      //etc..
    }
  }
  void setAutologin(bool b) {
    autoLogin_ = b;
  }
  /// Blocking call that will wait until the character is logged in
  void waitUntilLoggedIn() {
    //Wait on some condition variable that notifies when we get some packet that indicates successful login
  }
};

class Bot {
public:
  void handlePacket(const Packet &packet) {

  }
}

int main() {
  {
    LoginModule lm;
    lm.setAutologin(true);
    lm.handlePacket(Packet(Packet::Opcode::kServerListing));
    lm.handlePacket(Packet(Packet::Opcode::kClientCafe));
  }
  {
    LoginModule lm;
    lm.setAutologin(false);
    lm.handlePacket(Packet(Packet::Opcode::kServerListing));
    lm.handlePacket(Packet(Packet::Opcode::kClientCafe));
  }
  return 0;
}