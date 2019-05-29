### Here are some random thoughts

Is there any benefit to subscribing to a particular player's movement?

If some of this code is going to be built to send packets and wait for responses, we're going to need a timeout concept.

Some important data structures:
- A data structure for nearby mobs
  - Find mobs based on a location, % hp, level, type (champ, pt, pt champ, etc)
- A data structure for nearby players
- A data structure for nearby items on the ground

We're going to need other general blocking calls. For example, wait for all data to be received after login

Wrap Atomic Controllers in namespaces

Maybe its a good idea to use the Template Method design pattern for character logic