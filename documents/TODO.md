# TODO

- Delete "old_config" from protobuf definition & UI
- Address ASAN-detected memory leaks
  - `sro::navmesh::triangulation::NavmeshTriangulation::buildNavmeshForRegion`
- eventBroker.cpp:130 handle memory leak when destructed, events lost in handle function
- Get away from include & src directories in silkroad_lib
- It would be really nice to have a local list of known characters
- Add UI element to toggle packet logging to console
- Kill clients on shutdown
- Auto relogin when a character disconnects