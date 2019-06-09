### Welcome.

# Abstract

- The `Loader` launches the sro_client.exe process, suspends the process, injects loaderDll.dll, then resumes the process.
- The `Proxy` is started; ready to receive tcp connections.
- The injected dll causes the client to initiate a tcp connection to our proxy rather than the actual gameserver.
- The `Proxy` can forward any traffic to/from the client/server.
- The `Proxy` can block any traffic to the client or server.
- The `Proxy` can inject indistinguishable packets to the client or server.
- The `Bot` has a real-time view into the packet stream between client and server.
- The `Bot` can insert, remove, and modifiy any packet in the stream.
- The `Bot` has the interface to theoretically be able to do anything any human player can do.

# Code view

#### Lets dive in, first looking at the entrypoint: `phConnector.cpp` (insert gratitudes).
- We instantiate a single `Session` and then start the session. For now, `Session::start()` will be a blocking call, so we can only run 1 session.
- `Session::start()` starts the loader and the proxy.
- `Loader::startClient()` launches the sro_client.exe process, suspends the process, injects loaderDll.dll, then resumes the process.
- `Proxy::start()` constantly runs `boost::io_service`s where `Proxy::HandleAccept` will be called for one connection from the game client at a time.
- A secure connection is first established to the GatewayServer. The client asks for the ip address of the game server, known as the AgentServer, then establishes a secure connection.
- I believe `ServerConnection::Connect()` is a blocking call.

#### Now, we'll look at the important objects and their roles.
- A `Session` contains a `BrokerSystem`, a `Proxy`, and a `Bot`.
- The `Bot` and `Proxy` are constructed each with a reference to the `BrokerSystem`.
- The `BrokerSystem` has two main "channels": ClientPacket and ServerPacket.
- On the `Bot`'s construction, it subscribes to any interesting packet opcodes on either the ClientPacket or ServerPacket channel by giving a `PacketHandleFunction`.
- On the `Proxy`'s construction, it gives the `BrokerSystem` a pointer to `Proxy::inject`.
- As the `Proxy` runs and packets are received, they're passed to `BrokerSystem::packetReceived` for the broker to distribute to all subscribed parties.
- Once the `Bot`'s `PacketHandleFunction` is called with a pointer to a `PacketParser`, it tries to `dynamic_cast` it to an inheriting class of `PacketParser`(ex. `ClientChatPacket`).
- `PacketParser`s are a lazy-eval objectified wrapper around any packet. If N `PacketHandleFunction`s access the data in a packet, the packet data parsed into the object 0 times if N==0 or 1 time if N>0. `PacketParser`s always have N space overhead, where N is the size of the data in the packet(after the opcode).
- The user of an inherited class of `PacketParser`, such as `ClientChatPacket *clientChat` can directly access data of the packet in an object-oriented way via `clientChat->message()`.
- If the user needs to send a packet on a channel, this is easily done by `PacketBuilder` objects. Inheriting objects of `PacketBuilder` implement sufficient constructors for each of the valid packet configurations. Then, calling `packet()` on one of these inheriting objects, a complete packet is built and returned. Finally, the packet is passed to `BrokerSystem::injectPacket` and after many levels of abstraction, it's essentially pushed into the outgoing queue of tcp messages to be sent to either the client or the server.
- Right before the `PacketHandleFunction` exits, it returns `true` if the packet should be allowed to pass and `false` otherwise.
- To assist with packet building, and using from the `Bot` side, the `PacketEnums` namespace exists.
