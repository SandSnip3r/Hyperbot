# Hyperbot

Hyperbot is a bot for Silkroad Online. Is is built a bit differently than typical Silkroad bots because it serves a different purpose from typical Silkroad bots. Hyperbot is first and foremost a generic platform to automate arbitrary actions in Silkroad Online. Currently, Hyperbot is exclusively used for deep reinforcement learning.

## Components

Hyperbot is comprised of 3 separate components:

1. Hyperbot itself. This is the core bot. This must be run in Linux on a machine with a GPU.
2. ClientManager. This part launches/kills Silkroad game clients. Silkroad clients can only be launched on Windows, so this part must be run on Windows. If clientless is used, this part does not need to be running.
3. The user interface. Currently, the user interface is the only way to control deep reinforcement learning training. The UI dynamically reconnects and so it can be run after Hyperbot has been started and it can be closed after training has been initiated.

## Building

### ClientManager

Since ClientManager must be run on Windows, we must also build it in Windows. Below I will outline the steps to build ClientManager on Windows using Ubuntu24 in WSL 1 (primarily for the bash command line).

1. Install Visual Studio Community 2022. We will use CMake and VCPkg from this install.

_Note: Visual Studio version probably does not matter much. It might be possible to use a newer one if it exists._

2. Set the following in your WSL 1's Ubuntu `.bashrc`:
```
alias CMake='/mnt/c/Program\ Files/Microsoft\ Visual\ Studio/2022/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'
export VCPKG_ROOT="/mnt/c/Program Files/Microsoft Visual Studio/2022/Community/VC/vcpkg/"
export PATH="$VCPKG_ROOT:$PATH"
```

_Note: Your Visual Studio install path might be different._

3. Install dependencies using vcpkg:
```
cd Hyperbot
vcpkg.exe install --triplet=x86-windows
```