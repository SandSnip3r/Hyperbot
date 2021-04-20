#include "sharedMemoryWriter.hpp"

#include <stdexcept>

std::string GetLastErrorAsString() {
  //Get the error message, if any.
  DWORD errorMessageID = ::GetLastError();
  if(errorMessageID == 0) {
    return std::string(); //No error message has been recorded
  }

  LPSTR messageBuffer = nullptr;
  size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                               NULL, errorMessageID, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&messageBuffer, 0, NULL);

  std::string message(messageBuffer, size);

  //Free the buffer.
  LocalFree(messageBuffer);

  return message;
}

SharedMemoryWriter::SharedMemoryWriter(const std::string &name, int fileSize) : fileName_(name), fileSize_(fileSize) {
  hMapFile_ = CreateFileMapping(INVALID_HANDLE_VALUE,    // use paging file
                                NULL,                    // default security
                                PAGE_READWRITE,          // read/write access
                                0,                       // maximum object size (high-order DWORD)
                                fileSize_,               // maximum object size (low-order DWORD)
                                fileName_.c_str());      // name of mapping object

  if (hMapFile_ == NULL) {
    throw std::runtime_error("Could not create file mapping object " + GetLastErrorAsString());
  }
}

SharedMemoryWriter::~SharedMemoryWriter() {
  if (hMapFile_ != NULL) {
    CloseHandle(hMapFile_);
  }
}

SharedMemoryWriter::MapView::MapView(HANDLE fileHandle, int fileSize) {
  if (fileHandle == NULL) {
    throw std::runtime_error("Trying to write to file that does not exist");
  }

  // Map file
  buffer_ = (LPTSTR) MapViewOfFile(fileHandle,          // handle to map object
                                    FILE_MAP_ALL_ACCESS, // read/write permission
                                    0,
                                    0,
                                    fileSize);

  if (buffer_ == NULL) {
    throw std::runtime_error("Could not map view of file " + GetLastErrorAsString());
  }
}

SharedMemoryWriter::MapView::~MapView() {
  // Unmap file
  UnmapViewOfFile(buffer_);
}

PVOID SharedMemoryWriter::MapView::getBuffer() {
  return buffer_;
}