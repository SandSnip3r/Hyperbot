# Transcending To Become Multi-Character

This doc contains considerations made when evolving Hyperbot to control multiple characters concurrently.

## WorldState

Where should the WorldState should live?

Should there be one WorldState shared between all sessions?

If all entities are stored in the world state, then sessions will have access to information that they normally would not have. In my original use-case, which was collaborative botting, this is great. We can see things like the inventory of all characters controlled by the bot. In the case of 1v1 PVP, this is not ideal, because without explicit avoidance, hidden state will be available.

What if we want a combination? For example, party vs party pvp.

### Conclusion

It seems that we need some "team" concept where a team shares a WorldState.

## Config

I hate my current config set up...

There is some global configuration, such as client path.
There is some "session"-specific configuration, such as each character config.
We don't want to worry about multiple sessions writing to the same single config.

What about startup behavior? Which characters should we launch? Or what accounts should we log into? For now, I don't think we need to care.

### Conclusion

These configs should be separate files.
- One global config file
- Many character config files, one for each character

## EventBroker

How should we be using EventBroker?

These are the current classes which subscribe to EventBroker:

- Bot
- StatAggregator
- Self
- UserInterface
- Hyperbot

Most events are specific to a single character. Just a few are global, like config changes.