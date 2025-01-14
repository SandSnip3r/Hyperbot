#pragma once

#include <filesystem>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#endif

std::filesystem::path getAppDataPath();

namespace edxLabs {

#if defined(_WIN32)

// Injects a DLL into a process at the specified address
BOOL InjectDLL(HANDLE hProcess, const char * dllNameToLoad, const char * funcNameToLoad, DWORD injectAddress, bool bDebugAttach);

// Returns the entry point of an EXE
ULONGLONG GetEntryPoint(const char * filename);

// Returns a pointer to a hostent object for the specified address
hostent * GetHost(const char * address);

// Creates a suspended process
bool CreateSuspendedProcess(const std::string & filename, const std::string & fileargs, STARTUPINFOA & si, PROCESS_INFORMATION & pi);

// Returns true if the host is accessible
bool CanGetHostFromAddress(const std::string & address);

// Returns the writable directory for this framework.
// Type in "%appdata%/edxLabs" to access the directory.
std::string GetWriteableDirectory(std::string baseDir);

// Creates a codecave
BOOL CreateCodeCave(DWORD destAddress, BYTE patchSize, VOID (*function)(VOID));

// Patches bytes in the current process
BOOL WriteBytes(DWORD destAddress, LPVOID patch, DWORD numBytes);

// Reads bytes in the current process
BOOL ReadBytes(DWORD sourceAddress, LPVOID buffer, DWORD numBytes);

// Reads bytes of a process
BOOL ReadProcessBytes(HANDLE hProcess, DWORD destAddress, LPVOID buffer, DWORD numBytes);

// Writes bytes to a process
BOOL WriteProcessBytes(HANDLE hProcess, DWORD destAddress, LPVOID patch, DWORD numBytes);

// Creates a console, need to call FreeConsole before exit
VOID CreateConsole(CONST CHAR * winTitle);

// Forward declaration for the FileChooser class data so it is not exposed.
struct tFileChooserData;

// File chooser class
class FileChooser {
public:
  FileChooser();
  ~FileChooser();

  // Sets the initial directory the file chooser looks in.
  void SetInitialDirectory(const char * pDir);

  // Sets the default dialog title of the file chooser component.
  void SetDialogTitle(const char * pTitle);

  // Sets the default filename in the file choose dialog.
  void SetDefaultFileName(const char * pFileName);

  // Adds a file browsing filter.
  void AddFilter(const char * pFilterName, const char * pFilterExt);

  // Allow the user to select a file. (open => true for param, save => false for param)
  bool ShowChooseFile(bool open);

  // Returns the file path of the selected file.
  const char * GetSelectedFilePath();

  // Returns the file directory of the selected file.
  const char * GetSelectedFileDirectory();

  // Returns the filename of the selected file.
  const char * GetSelectedFileName();

  // Returns the file title of the selected file.
  const char * GetSelectedFileTitle();

  // Returns the file extension of the selected file.
  const char * GetSelectedFileExtension();

private:
  tFileChooserData* data;
};

class ConfigFile {
public:
  ConfigFile();
  ~ConfigFile();

  // Opens a file to work with
  void Open(std::string filename, bool useCurrentPath, bool & fileExists);

  // Set the section that the 'Write' and 'Read' functions use
  void SetSection(const std::string & section);

  // Get the section that the 'Write' and 'Read' functions use
  std::string GetSection() const;

  // Writes to the current section
  void Write(const std::string & key, const std::string & data);

  // Writes to any section
  void WriteTo(const std::string & section, const std::string & key, const std::string & data);

  // Read from the current section
  std::string Read(const std::string & key);

  // Read from any section
  std::string ReadFrom(const std::string & section, const std::string & key);
private:
  std::string mFileName;
  std::string mSection;
};

// Returns an integer from a hex string
int HexStringToInteger(const std::string & str);

// Tokenizes a string into a vector
std::vector<std::string> TokenizeString(const std::string& str, const std::string& delim);

// Byte signature searching function
std::vector<LONGLONG> FindSignature(LPBYTE sigBuffer, LPBYTE sigWildCard, DWORD sigSize, PBYTE pBuffer, LONGLONG size);

// The typedef for the user defined ProcessExe function
// void DefaultFunction(HANDLE hFile, PBYTE pMappedFileBase, DWORD dwFileSizeLow, DWORD dwFileSizeHigh)
typedef void (*ProcessFileFunc)(HANDLE, PBYTE, DWORD, DWORD);

// This function will memory map the specified file and call the userFunction with the data and size
void ProcessFile(const char * filename, ProcessFileFunc userFunction);

class ProcessFileObject {
public:
  // A handle to the file
  HANDLE hFile;

  // The file mapping object used to read from the file
  HANDLE hFileMapping;

  // A pointer to the bytes of the file
  PBYTE pMappedFileBase;

  // File size
  DWORD dwSizeLow;
  DWORD dwSizeHigh;

  ProcessFileObject();
  ~ProcessFileObject();
  bool Open(const char * filename);
  void Close();

  // A reinterpretation of the data stream if the file is executable
  PIMAGE_NT_HEADERS GetNTHeader();
};

#endif

} // namespace edxLabs