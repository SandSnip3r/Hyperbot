template<typename IndexType>
void SingleRegionNavmeshTriangulationState<IndexType>::setObjectData(const ObjectData &objectData) {
  objectData_ = objectData;
}

template<typename IndexType>
bool SingleRegionNavmeshTriangulationState<IndexType>::isOnObject() const {
  return objectData_.has_value();
}

template<typename IndexType>
void SingleRegionNavmeshTriangulationState<IndexType>::setOnTerrain() {
  objectData_.reset();
}

template<typename IndexType>
const ObjectData& SingleRegionNavmeshTriangulationState<IndexType>::getObjectData() const {
  if (!objectData_) {
    throw std::logic_error("Asking for object data for state that isnt on object");
  }
  return objectData_.value();
}

template<typename IndexType>
bool SingleRegionNavmeshTriangulationState<IndexType>::isSameTriangleAs(const SingleRegionNavmeshTriangulationState<IndexType> &otherState) const {
  if (static_cast<pathfinder::navmesh::NavmeshAStarState<IndexType>>(*this).isSameTriangleAs(static_cast<pathfinder::navmesh::NavmeshAStarState<IndexType>>(otherState))) {
    return (objectData_ == otherState.objectData_);
  } else {
    return false;
  }
}

template<typename IndexType>
bool SingleRegionNavmeshTriangulationState<IndexType>::isTraversingLink() const {
  return linkId_.has_value();
}

template<typename IndexType>
void SingleRegionNavmeshTriangulationState<IndexType>::resetLinkId() {
  linkId_.reset();
}

template<typename IndexType>
void SingleRegionNavmeshTriangulationState<IndexType>::setLinkId(const IndexType id) {
  linkId_ = id;
}

template<typename IndexType>
IndexType SingleRegionNavmeshTriangulationState<IndexType>::getLinkId() const {
  if (!linkId_) {
    throw std::logic_error("Asking for link ID for state that is not traversing a link");
  }
  return linkId_.value();
}

template<typename IndexType>
bool operator==(const SingleRegionNavmeshTriangulationState<IndexType> &s1, const SingleRegionNavmeshTriangulationState<IndexType> &s2) {
  if (static_cast<pathfinder::navmesh::NavmeshAStarState<IndexType>>(s1) == static_cast<pathfinder::navmesh::NavmeshAStarState<IndexType>>(s2)) {
    // Do check on derived data
    if (s1.isOnObject() && s2.isOnObject()) {
      // Both are on objects, compare these objects
      return s1.getObjectData() == s2.getObjectData();
    } else if (s1.isOnObject() == s2.isOnObject()) {
      // Both must be not on objects; nothing else to compare
      return true;
    } else {
      // One is on an object, other is not
      return false;
    }
  } else {
    // Base classes arent equal, no need to evaluate data from derived class
    return false;
  }
}

template<typename IndexType>
bool operator<(const SingleRegionNavmeshTriangulationState<IndexType> &s1, const SingleRegionNavmeshTriangulationState<IndexType> &s2) {
  if (static_cast<pathfinder::navmesh::NavmeshAStarState<IndexType>>(s1) == static_cast<pathfinder::navmesh::NavmeshAStarState<IndexType>>(s2)) {
    // Do check on derived data
    if (s1.isOnObject() && s2.isOnObject()) {
      // Both are on objects, compare these objects
      return (s1.getObjectData() < s2.getObjectData());
    } else {
      // No other data to compare, just return the difference between the existence of object data
      return (s1.isOnObject() ? 1:0) < (s2.isOnObject() ? 1:0);
    }
  } else {
    // Base classes arent equal, use their comparison function
    return (static_cast<pathfinder::navmesh::NavmeshAStarState<IndexType>>(s1) < static_cast<pathfinder::navmesh::NavmeshAStarState<IndexType>>(s2));
  }
}

template<typename IndexType>
std::ostream& operator<<(std::ostream& stream, const SingleRegionNavmeshTriangulationState<IndexType> &state) {
  constexpr const bool kPrintIndicesAsGlobal{false};
  if (state.isGoal()) {
    stream << "[GOAL]";
  } else if (!state.hasEntryEdgeIndex()) {
    stream << "[START]";
  }
  stream << "(tri ";
  if constexpr (kPrintIndicesAsGlobal) {
    stream << state.getTriangleIndex();
  } else {
    stream << (state.getTriangleIndex()>>16) << ',' << (state.getTriangleIndex()&0xFFFF);
  }

  stream << "), (edge ";
  if (state.hasEntryEdgeIndex()) {
    if constexpr (kPrintIndicesAsGlobal) {
      stream << state.getEntryEdgeIndex();
    } else {
      stream << (state.getEntryEdgeIndex()>>16) << ',' << (state.getEntryEdgeIndex()&0xFFFF);
    }
  } else {
    stream << '_';
  }

  stream << "), (obj ";
  if (state.isOnObject()) {
    stream << ',' << state.getObjectData().objectInstanceId << ',' << state.getObjectData().objectAreaId;
  } else {
    stream << '_';
  }

  stream << "), (link ";
  if (state.isTraversingLink()) {
    stream << state.getLinkId();
  } else {
    stream << '_';
  }
  stream << ")";
  return stream;
}