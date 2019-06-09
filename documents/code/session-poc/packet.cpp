#include "packet.hpp"

Packet::Packet(Opcode code) : opcode_(code) {

}

Packet::Opcode Packet::opcode() const {
  return opcode_;
}

int Packet::hpIncreaseAmount() const {
  return hpIncreaseAmount_;
}

int Packet::hp() const {
  return hp_;
}

int Packet::maxHp() const {
  return maxHp_;
}

void Packet::setHpIncreaseAmount(int amnt) {
  hpIncreaseAmount_ = amnt;
}

void Packet::setHp(int hp) {
  hp_ = hp;
}

void Packet::setMaxHp(int hp) {
  maxHp_ = hp;
}

namespace PacketPrinting {

std::string opcodeToStr(Packet::Opcode opcode) {
  switch (opcode) {
    case Packet::Opcode::kServerCharacter:
      return "kServerCharacter";
    case Packet::Opcode::kServerHpMpUpdate:
      return "kServerHpMpUpdate";
    case Packet::Opcode::kClientItemUse:
      return "kClientItemUse";
    default:
      return "unknown";
  }
}

} // namespace PacketPrinting