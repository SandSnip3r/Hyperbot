#include "rl/observationAndActionStorage.hpp"

#include <absl/log/log.h>

namespace rl {

ObservationAndActionStorage::ObservationAndActionStorage(size_t capacity) : capacity_(capacity) {}

std::pair<ObservationAndActionStorage::Id, std::vector<ObservationAndActionStorage::Id>> ObservationAndActionStorage::addObservationAndAction(common::PvpDescriptor::PvpId pvpId, const std::string &intelligenceName, const Observation &observation, std::optional<int> actionIndex) {
  std::unique_lock lock{mutex_};
  std::vector<Id> deletedIds;
  if (size() == capacity_) {
    // Buffer is full, need to delete the oldest PVP.
    if (pvpIdAndIntelligenceNameDeque_.empty()) {
      throw std::runtime_error("ObservationAndActionStorage: Buffer is full but no PVPs to delete.");
    }
    auto [oldestPvp, oldestIntelligenceName] = pvpIdAndIntelligenceNameDeque_.front();
    pvpIdAndIntelligenceNameDeque_.pop_front();
    auto intelligenceToBufferMapIt = buffer_.find(oldestPvp);
    if (intelligenceToBufferMapIt == buffer_.end()) {
      throw std::runtime_error("ObservationAndActionStorage: Oldest PVP not found in buffer.");
    }
    auto bufferIt = intelligenceToBufferMapIt->second.find(oldestIntelligenceName);
    if (bufferIt == intelligenceToBufferMapIt->second.end()) {
      throw std::runtime_error("ObservationAndActionStorage: Oldest intelligence name not found in buffer.");
    }
    const auto &observations = bufferIt->second;
    // LOG(INFO) << "At capacity, deleting oldest PVP: " << oldestPvp << ", intelligence name: " << oldestIntelligenceName << ", size: " << observations.size();
    for (size_t i=0; i<observations.size(); ++i) {
      // Look up id for buffer indices.
      BufferIndices bufferIndices(oldestPvp, oldestIntelligenceName, i);
      auto idIt = indexToIdMap_.find(bufferIndices);
      if (idIt == indexToIdMap_.end()) {
        throw std::runtime_error("ObservationAndActionStorage: ID not found in buffer.");
      }
      deletedIds.push_back(idIt->second);
      indexToIdMap_.erase(idIt);
    }
    intelligenceToBufferMapIt->second.erase(bufferIt);
    for (Id id : deletedIds) {
      idToIndexMap_.erase(id);
    }
  }

  std::deque<ObservationAndActionType> &observations = buffer_[pvpId][intelligenceName];
  observations.push_back({observation, actionIndex});
  size_t index = observations.size() - 1;
  if (index == 0) {
    // LOG(INFO) << "First of a new pvp added, pushing to deque: " << pvpId << ", intelligence name: " << intelligenceName;
    pvpIdAndIntelligenceNameDeque_.emplace_back(pvpId, intelligenceName);
  }
  Id id = nextId_;
  if (id == std::numeric_limits<Id>::max()) {
    throw std::runtime_error("ObservationAndActionStorage: ID overflow.");
  }
  ++nextId_;
  BufferIndices bufferIndices(pvpId, intelligenceName, index);
  idToIndexMap_[id] = bufferIndices;
  indexToIdMap_[bufferIndices] = id;
  // LOG(INFO) << "Added observation and action to storage. ID: " << id << ", PVP ID: " << pvpId << ", Intelligence Name: " << intelligenceName << ", Action Index: " << index << ", Size: " << size() << ", Deleted IDs: " << deletedIds.size();
  return {id, deletedIds};
}

const ObservationAndActionStorage::ObservationAndActionType& ObservationAndActionStorage::getObservationAndAction(Id id) const {
  std::unique_lock lock{mutex_};
  auto it = idToIndexMap_.find(id);
  if (it == idToIndexMap_.end()) {
    throw std::out_of_range("getObservationAndAction: ID not found in buffer.");
  }
  const BufferIndices &index = it->second;
  auto it1 = buffer_.find(index.pvpId);
  if (it1 == buffer_.end()) {
    throw std::out_of_range("Index(pvpId) not found in buffer.");
  }
  auto it2 = it1->second.find(index.intelligenceName);
  if (it2 == it1->second.end()) {
    throw std::out_of_range("Index(intelligenceName) not found in buffer.");
  }
  const auto& observations = it2->second;
  if (index.actionIndex >= observations.size()) {
    throw std::out_of_range("Index out of range in transition vector.");
  }
  return observations.at(index.actionIndex);
}

bool ObservationAndActionStorage::hasPrevious(Id id) const {
  std::unique_lock lock{mutex_};
  auto it = idToIndexMap_.find(id);
  if (it == idToIndexMap_.end()) {
    throw std::out_of_range("hasPrevious: ID not found in buffer.");
  }
  const BufferIndices &index = it->second;
  return index.actionIndex > 0;
}

ObservationAndActionStorage::Id ObservationAndActionStorage::getPreviousId(Id id) const {
  std::unique_lock lock{mutex_};
  auto it = idToIndexMap_.find(id);
  if (it == idToIndexMap_.end()) {
    throw std::out_of_range("getPrevious: ID not found in buffer.");
  }
  const BufferIndices &index = it->second;
  if (index.actionIndex == 0) {
    throw std::out_of_range("No previous observation.");
  }
  BufferIndices previousIndex(index.pvpId, index.intelligenceName, index.actionIndex - 1);
  auto it1 = indexToIdMap_.find(previousIndex);
  if (it1 == indexToIdMap_.end()) {
    throw std::out_of_range("Previous ID not found in buffer.");
  }
  return it1->second;
}

ObservationAndActionStorage::Id ObservationAndActionStorage::getNextId(Id id) const {
  std::unique_lock lock{mutex_};
  auto it = idToIndexMap_.find(id);
  if (it == idToIndexMap_.end()) {
    throw std::out_of_range("getNextId: ID not found in buffer.");
  }
  const BufferIndices &index = it->second;
  BufferIndices nextIndex(index.pvpId, index.intelligenceName, index.actionIndex + 1);
  auto it1 = indexToIdMap_.find(nextIndex);
  if (it1 == indexToIdMap_.end()) {
    throw std::out_of_range("Next ID not found in buffer.");
  }
  return it1->second;
}

size_t ObservationAndActionStorage::size() const {
  size_t totalSize = 0;
  for (const auto &pvpPair : buffer_) {
    for (const auto &intelligencePair : pvpPair.second) {
      totalSize += intelligencePair.second.size();
    }
  }
  return totalSize;
}

template<typename T>
static void writeBinary(std::ostream &out, const T &value) {
  out.write(reinterpret_cast<const char *>(&value), sizeof(T));
}

template<typename T>
static void readBinary(std::istream &in, T &value) {
  in.read(reinterpret_cast<char *>(&value), sizeof(T));
}

void ObservationAndActionStorage::saveToStream(std::ostream &out) const {
  std::unique_lock lock{mutex_};
  writeBinary(out, capacity_);
  writeBinary(out, nextId_);
  size_t pvpCount = buffer_.size();
  writeBinary(out, pvpCount);
  for (const auto &[pvpId, intelligenceMap] : buffer_) {
    writeBinary(out, pvpId);
    size_t intCount = intelligenceMap.size();
    writeBinary(out, intCount);
    for (const auto &[name, deque] : intelligenceMap) {
      uint32_t nameLen = name.size();
      writeBinary(out, nameLen);
      out.write(name.data(), nameLen);
      size_t obsCount = deque.size();
      writeBinary(out, obsCount);
      for (size_t i = 0; i < deque.size(); ++i) {
        BufferIndices idx(pvpId, name, i);
        Id id = indexToIdMap_.at(idx);
        writeBinary(out, id);
        deque[i].observation.saveToStream(out);
        bool hasAct = deque[i].actionIndex.has_value();
        writeBinary(out, hasAct);
        if (hasAct) {
          int32_t act = deque[i].actionIndex.value();
          writeBinary(out, act);
        }
      }
    }
  }
  size_t dequeSize = pvpIdAndIntelligenceNameDeque_.size();
  writeBinary(out, dequeSize);
  for (const auto &[pvpId, name] : pvpIdAndIntelligenceNameDeque_) {
    writeBinary(out, pvpId);
    uint32_t nameLen = name.size();
    writeBinary(out, nameLen);
    out.write(name.data(), nameLen);
  }
}

void ObservationAndActionStorage::loadFromStream(std::istream &in) {
  std::unique_lock lock{mutex_};
  size_t capacity;
  readBinary(in, capacity);
  if (capacity != capacity_) {
    throw std::runtime_error("ObservationAndActionStorage capacity mismatch");
  }
  readBinary(in, nextId_);
  buffer_.clear();
  idToIndexMap_.clear();
  indexToIdMap_.clear();
  size_t pvpCount;
  readBinary(in, pvpCount);
  for (size_t p = 0; p < pvpCount; ++p) {
    common::PvpDescriptor::PvpId pvpId;
    readBinary(in, pvpId);
    size_t intCount;
    readBinary(in, intCount);
    for (size_t j = 0; j < intCount; ++j) {
      uint32_t nameLen;
      readBinary(in, nameLen);
      std::string name(nameLen, '\0');
      in.read(name.data(), nameLen);
      size_t obsCount;
      readBinary(in, obsCount);
      std::deque<ObservationAndActionType> &dq = buffer_[pvpId][name];
      dq.resize(obsCount);
      for (size_t i = 0; i < obsCount; ++i) {
        Id id;
        readBinary(in, id);
        dq[i].observation = Observation::loadFromStream(in);
        bool hasAct;
        readBinary(in, hasAct);
        if (hasAct) {
          int32_t act;
          readBinary(in, act);
          dq[i].actionIndex = act;
        } else {
          dq[i].actionIndex.reset();
        }
        BufferIndices idx(pvpId, name, i);
        idToIndexMap_[id] = idx;
        indexToIdMap_[idx] = id;
      }
    }
  }
  pvpIdAndIntelligenceNameDeque_.clear();
  size_t dequeSize;
  readBinary(in, dequeSize);
  for (size_t i = 0; i < dequeSize; ++i) {
    common::PvpDescriptor::PvpId pvpId;
    readBinary(in, pvpId);
    uint32_t nameLen;
    readBinary(in, nameLen);
    std::string name(nameLen, '\0');
    in.read(name.data(), nameLen);
    pvpIdAndIntelligenceNameDeque_.emplace_back(pvpId, name);
  }
}

} // namespace rl