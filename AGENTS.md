# Hyperbot Agent Guidelines

Hyperbot is a toolkit for automating the MMORPG **Silkroad Online**.  It uses a
custom loader DLL and packet proxy to control one or more game clients from a
single host process.  The core framework is written in C++ and built with
CMake, pulling dependencies via **vcpkg**, while Python modules provide
reinforcement learning logic.

## Code Style

- Use **2 spaces** for indentation (no tabs).
- Private members often use a trailing underscore (`foo_`).
- `.hpp`/`.cpp` files are preferred for headers and sources.
- Avoid modifying anything under `third_party/` unless necessary.

## Design Philosophy

- Hyperbot aims to be an extensible platform for automating Silkroad Online.
- Components are separated into loader, proxy, bot logic, and user interfaces.
- Clean abstractions are favored so new features or RL algorithms can be added
  with minimal coupling.

## Project Structure

- `bot/` – core bot source code and tests
- `client_manager/` – launches and controls game clients on Windows
- `loader_dll/` – DLL injected into the game client for connection redirection
- `silkroad_lib/` – shared utilities and game-specific data types
- `rl_ui/` – Qt application for reinforcement learning monitoring
- `ui/` – legacy UI code and widgets *(deprecated; do not modify)*
- `ui_proto/` – protobuf definitions shared between components
- `tools/` – helper scripts for packet parsing and state machine creation
- `third_party/` – vendored libraries; keep changes here to a minimum
- `documents/` – design notes and assorted documentation

No build or test commands are provided here as the setup is fairly involved.


