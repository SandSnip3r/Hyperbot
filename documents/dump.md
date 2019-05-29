Is there any benefit to subscribing to a particular player's movement?

If some of this code is going to be built to send packets and wait for responses, we're going to need a timeout concept.

Need:
A data structure for nearby mobs
A data structure for nearby players
A data structure for nearby items on the ground

We're going to need other general blocking calls
- Wait for all data to be received after login

Wrap Atomic Controllers in namespaces