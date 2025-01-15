# TODO

- Delete "old_config" from protobuf definition & UI
- Address ASAN-detected memory leaks
  - `sro::navmesh::triangulation::NavmeshTriangulation::buildNavmeshForRegion`
- eventBroker.cpp:130 handle memory leak when destructed, events lost in handle function