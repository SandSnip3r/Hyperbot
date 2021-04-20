#ifndef SHARED_MEMORY_WRITER_HPP_
#define SHARED_MEMORY_WRITER_HPP_

#define WIN32_LEAN_AND_MEAN

#include <windows.h>

#include <stdexcept>
#include <string>

std::string GetLastErrorAsString();

class SharedMemoryWriter {
public:
  SharedMemoryWriter(const std::string &name, int fileSize);
  ~SharedMemoryWriter();
  void seek(int pos) { dataIndexBytes_ = pos; }

  template<typename T>
  void writeData(const T &data) {
    if (sizeof(T) > fileSize_) {
      throw std::runtime_error("Trying to write data type that is too large for file");
    }

    MapView view(hMapFile_, fileSize_);

    CopyMemory((uint8_t*)view.getBuffer()+dataIndexBytes_, &data, sizeof(data));

    dataIndexBytes_ += sizeof(data);
  }
  
private:
  HANDLE hMapFile_{NULL};
  const std::string fileName_;
  const int fileSize_;
  int dataIndexBytes_{0};

  class MapView {
  public:
    MapView(HANDLE fileHandle, int fileSize);
    ~MapView();
    PVOID getBuffer();
  private:
    PVOID buffer_;
  };
};

#undef WIN32_LEAN_AND_MEAN

#endif // SHARED_MEMORY_WRITER_HPP_