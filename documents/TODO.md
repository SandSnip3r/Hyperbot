# TODO

- Delete "old_config" from protobuf definition & UI
- Address ASAN-detected memory leaks
  - `sro::navmesh::triangulation::NavmeshTriangulation::buildNavmeshForRegion`
- eventBroker.cpp:130 handle memory leak when destructed, events lost in handle function
- Get away from include & src directories in silkroad_lib
- It would be really nice to have a local list of known characters
- Add UI element to toggle packet logging to console
- Kill clients on shutdown
  - I should do this by detecting a missing heartbeat from the bot in the ClientManager
- Auto relogin when a character disconnects
- Ensure that no state machine does any work in its constructor
  - Bonus if we can guarantee it
- Remove StateMachineActiveTooLong event
- Remove StateMachine created/destroyed in ctors/dtors
- Make StateMachine::name a virtual function
- Handle the case when state::machine::PickItem's target item is picked by someone else