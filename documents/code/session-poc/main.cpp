#include "bot.hpp"
#include "brokerSystem.hpp"

#include <iostream>
#include <random>
#include <algorithm>
#include <unistd.h>

std::mt19937 createRandomEngine() {
  std::random_device rd;
  std::array<int, std::mt19937::state_size> seed_data;
  std::generate_n(seed_data.data(), seed_data.size(), std::ref(rd));
  std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
  return std::mt19937(seq);
}

class Proxy {
private:
  BrokerSystem &broker_;
  void handlePacket(const Packet &packet, Packet::Direction packetDirection) {
    bool packetShouldBeForwarded = broker_.packetReceived(packet, packetDirection);
  }
  int maxHp{100};
  int currentHp{100};
public:
  Proxy(BrokerSystem &broker) : broker_(broker) {
    broker_.setInjectionFunction(std::bind(&Proxy::inject, this, std::placeholders::_1, std::placeholders::_2));
  }
  void inject(const Packet &packet, Packet::Direction packetDirection) {
#ifdef SIM
    // Emulate the server here
    if (packet.opcode() == Packet::Opcode::kClientItemUse) {
      // Player used a potion
      int amnt = packet.hpIncreaseAmount();
      currentHp = std::min(maxHp, currentHp+amnt);
      // Send a packet from the server indicating that the player's hp has risen
      Packet p(Packet::Opcode::kServerHpMpUpdate);
      p.setHp(currentHp);
      usleep(250000);
      broker_.packetReceived(p, Packet::Direction::kServerToClient);
    }
  }
  void simulate() {
    auto eng = createRandomEngine();
    std::bernoulli_distribution hpChangeDist(0.25);
    std::uniform_int_distribution<int> hpDecreaseDist(1,maxHp);
    // Send a char data packet which includes total HP
    {
      Packet p(Packet::Opcode::kServerCharacter);
      p.setMaxHp(maxHp);
      broker_.packetReceived(p, Packet::Direction::kServerToClient);
    }
    while (true) {
      if (hpChangeDist(eng)) {
        // Reduce hp
        int decreaseAmount = std::min(currentHp-1, hpDecreaseDist(eng));
        std::cout << "[Sim] Hit character for " << decreaseAmount << '\n';
        currentHp -= decreaseAmount;
        // Create a server packet for this event
        Packet p(Packet::Opcode::kServerHpMpUpdate);
        p.setHp(currentHp);
        broker_.packetReceived(p, Packet::Direction::kServerToClient);
      }
      usleep(500000);
    }
#endif
  }
};

class Session {
public:
  BrokerSystem broker_;
  Proxy proxy_{broker_};
  Bot bot_{broker_};
  Session() {

  }
  void run() {
    proxy_.simulate();
  }
};

using namespace std;

int main() {
  Session session;
  session.run();
  return 0;
}