### Welcome.

# Low level knowledge

## Abstract technical low-level view
- The `Loader` launches the sro_client.exe process, suspends the process, injects loaderDll.dll, then resumes the process.
- The `Proxy` is started; ready to receive tcp connections.
- The injected dll causes the client to initiate a tcp connection to our proxy rather than the actual gameserver.
- The `Proxy` can forward any traffic to/from the client/server.
- The `Proxy` can also inject any traffic to be indistinguishable to the server.

## Code view
Lets first look at the entrypoint. `phConnector.cpp` (insert gratitudes) contains our main function.
- We instantiate a single `Session` and then start the session. For now, `Session::start()` will be a blocking call, so we can only run 1 session for now.
- `Session::start()` starts the loader and the proxy.
- `Loader::startClient()` launches the sro_client.exe process, suspends the process, injects loaderDll.dll, then resumes the process.
- `Proxy::start()` constantly runs `boost::io_service`s where `Proxy::HandleAccept` will be called for one connection at a time.
- Some security stuff is done to initiate the connection.
- We first connect to the AgentServer. The client is probably asking for which ip to connect to for the game, known as the GatewayServer.
- It looks like `ServerConnection::Connect()` is blocking?

# Higher dumb-bot level

## Abstract & Theory
- The `Bot` has a real-time view into the packet stream between client and server.
- The `Bot` can insert, remove, and modifiy any packet in the stream.
- The `Bot` has the interface to theoretically be able to do anything any human player can do.

## Code view
- Within a `Session`, a `Bot` is constructed with a reference to `Proxy::inject()`. Incoming packets inside the proxy are given to `Bot::packetRecevied()`.
- `Proxy::inject` provides a method to inject a packet either to the server to spoof client actions or to the client to spoof server.
- `Bot::packetReceived` is where the main "brain" will connect. This is current lacking and is one of the next places for improvement.
