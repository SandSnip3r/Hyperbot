# How to build

## Prerequisites

- Visual Studio
  - Only 2019 tested
- CMake
- Boost 1.67.0

## Using WSL command line

1. Create a build directory
   - `mkdir build`
   - `cd build`
2. Run the Windows cmake.exe
   - Example location: `C:\Program Files\CMake\bin\cmake.exe`
   - Pass a generator
      - `-G "Visual Studio 16 2019"`
   - Pass a platform
      - `-A Win32`
   - Pass the path to your vcpkg toolchain file
      - `-DCMAKE_TOOLCHAIN_FILE="[path to vcpkg]\scripts\buildsystems\vcpkg.cmake"`
   - Pass the path to the root `CMakeLists.txt`
      - `../`
   - (Optional) Pass a toolset
      - `-T host=x64`
   - Final command example: `/mnt/c/Program\ Files/CMake/bin/cmake.exe -G "Visual Studio 16 2019" -A Win32 -DCMAKE_TOOLCHAIN_FILE="[path to vcpkg]\scripts\buildsystems\vcpkg.cmake" ../`
3. Run the MSBuild.exe supplied with Visual Studio
   - Example location: `C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Current\Bin`
   - Pass the path to the Solution file created by CMake
      - Probably `Hyperbot.sln` in your current directory
   - Final command example: `/mnt/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2019/Community/MSBuild/Current/Bin/MSBuild.exe Hyperbot.sln`
   - Note that the build will create a Debug directory at the top level of the project, this is where Hyperbot.exe exists

## Using Visual Studio

1. Open Visual Studio
2. File -> Open -> CMake...
3. Select root `CMakeLists.txt`
4. Set configuration x86-Debug
   - This configuration is defined in `CMakeSettings.json`
5. Right click the root `CMakeLists.txt` and choose "Generate Cache for Hyperbot"
   - I think this runs Ninja to generate build files in the output directory defined in `CMakeSettings.json`
   - If there are files in this output directory already, it could cause issues in Visual Studio
6. Build -> Build All

## Building Hyperbot UI Using Qt Creator

Under the Projects tab, find the Build & Run settings. Under Initial Configuration of the Build section, add your `CMAKE_TOOLCHAIN_FILE` if necessary (is necessary for vcpkg builds). Under Build  Steps, under Targets, unselect "all". Select "hyperbot-ui" and "ui-proto". Under the Run section, change "Run configuration" to "hyperbot-ui".