#ifndef PACKET_HPP
#define PACKET_HPP

#include <string>

class Packet {
public:
  enum class Direction { kClientToServer, kServerToClient, kBotToServer, kBotToClient };
  enum class Opcode { kServerCharacter, kServerHpMpUpdate, kClientItemUse };
private:
  Opcode opcode_;
  int hpIncreaseAmount_;
  int hp_;
  int maxHp_;
public:
  Packet(Opcode code);
  Opcode opcode() const;
  int hpIncreaseAmount() const;
  int hp() const;
  int maxHp() const;
  void setHpIncreaseAmount(int amnt);
  void setHp(int hp);
  void setMaxHp(int hp);
};

namespace PacketPrinting {

std::string opcodeToStr(Packet::Opcode opcode);

} // namespace PacketPrinting

#endif // PACKET_HPP