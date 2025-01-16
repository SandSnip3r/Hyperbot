#include <filesystem>
#include <iostream>

using namespace std;

int main() {
  // Configuration
    // Config
  // Packet producer/destination
    // Proxy
  // Packet "event" manager
  // Timer "event" manager"
  // General "event" manager
    // EventManager
  // Packet handler
    // Bot
  const filesystem::path kConfigPath{"./config.ini"};
  Config config(kConfigPath);

  PacketBroker packetbroker;
  Proxy proxy(packetbroker, config.getProxyConfig());
  Bot bot(packetbroker, config.getBotConfig());

  thread botThread{&Bot::run, bot};
  // Run the proxy on our current thread
  proxy.run();
  botThread.join();
  return 0;
}