# TODO

- Delete "old_config" from protobuf definition & UI
- Address ASAN-detected memory leaks
  - `sro::navmesh::triangulation::NavmeshTriangulation::buildNavmeshForRegion`
- eventBroker.cpp:130 handle memory leak when destructed, events lost in handle function
- Use .cpp & .hpp in silkroad_lib
- Fix paths in tools/*.py
- It would be really nice to have a local list of known characters