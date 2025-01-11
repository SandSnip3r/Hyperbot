# Proposed Directories

## Structure

bot/
client_controller/
dll/
pathfinder/
silkroad_lib/
proto/
ui/

## Explanation

`bot` contains the core bot. This is the part which has the network proxy, all the character-control logic, etc. The bot may launch clients and also may do reinforcement learning in JAX; these each add different build constraints (see below).

`client_controller` contains an application which runs as a background process and starts/kills clients upon command.

`dll` contains the dll to be injected into sro_client.exe.

`pathfinder` contains Pathfinder, which is shared between `bot` and `ui`.

`silkroad_lib` contains code which is shared between `bot` and `ui`, such as pk2 parsing, entity types/structs, etc.

`proto` contains protobuf definitions for communication between `ui`, `bot`, and `client_controller`.

`ui` contains the Qt project for the Hyperbot user interface.

# Build Constraints

If a directory is not listed below, it will have no build constraints.

 - bot
   - If built on linux, include JAX stuff
   - If built on windows, include client controller
 - client_controller
   - Build as 32 bit on windows, since the sro_client.exe and dll are also 32 bit. See below about building a DLL injector as 64 bit
 - dll
   - Must be built as 32 bit on windows, because the sro_client.exe is also 32 bit

#### Comments from Drew about building the DLL injector as 64 bit

*With a 32-bit injected dll, you can have a 64-bit injector, but not vice versa for obvious reasons of a 32-bit process not being able to interop with a 64-bit process.*

*Using a 64-bit injector only has a few caveats with making sure certain API functions are used because 32-bit on windows on a x64 arch is achived through https://en.wikipedia.org/wiki/WoW64 so there's a few gotchas with paths and api functions. For example, using a 64-bit api function like VirtualQueryEx, you need to use the right sized memory struct, since a 32-bit process will still have 64-bit memory regions from the OS - https://stackoverflow.com/a/26768537*

# Commands

```
/mnt/c/Program\ Files/CMake/bin/cmake.exe --preset win32
/mnt/c/Program\ Files/CMake/bin/cmake.exe --preset x64
```