I am building a relatively complex UI to control a bot for a game. The UI will mainly be used for an endpoint to configure and view stats of multiple reinforcement learning training sessions. I would really appreciate if you could help me wireflow for this UI.

Here are some important architectural design decisions that will influence which things need to happen in which order:

- The actual bot application runs on a linux server, maybe on another host.
- The UI may also run on the linux server but is more likely running on Windows.
- The bot will likely launch and control multiple game clients.
- Launching game clients requires a 3rd application. This application MUST run on windows. For each "game client" the bot requires to open, this application will launch a windows process of the game client and inject a DLL for connection redirection.

Here are some requirements:

- The UI first needs to ask where to connect to the bot server.
- Once the UI knows how to connect to the bot server, we should save this in a config so that we don't need to ask later.
- The UI connects to the bot server.
- The UI will display what characters the bot knows about. These are characters that maybe we've logged in to before, but are not necessarily logged in to now.
- The UI has a form which allows the initialization of a reinforcement learning training session.
  - The user picks two characters who will fight against each other.
  - The user picks a location on the map for the training to run.
  - The user then clicks "Run".
- For each training session that's running, there should be a UI that gives a nice summary of the state of the session.
  - Number of episodes
  - Number of actions
  - Some plots showing live stats
    - Rate of actions & episodes
- The UI (later in development) will have a read-only access version (so that others might connect and monitor the training statuses)
  - Maybe we check by authentication when the UI tries to connect to the bot server

Could you please give me a high level basic wireflow? Then, if necessary, i'll ask details.