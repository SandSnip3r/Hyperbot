# Hyperbot
This is a bot to automate everything a human player can do in an online MMORPG called [Silkroad Online](http://www.joymax.com/silkroad/).

## Silkroad
Silkroad Online is a historical fantasy MMORPG from Joymax based on the history of China along the famous Silk Road. Reproducing 7th century Silk Road trading while adding fantastic elements, Silkroad Online allows players to create their own characters, complete quests, fight monsters, level up, obtain pets, craft items, engage in PvP, and more.

The official Silkroad Online game has had a shifting in community with the introduction of pay-to-win strategies. I think the level 110 cap was the best level cap because of the effort required to get a full Egyptian set with weapons & accessories. The good news is that the private server community is huge. Different versions of the gameservers have been released and we can host our own local 110 level cap Silkroad server. There is also a pretty big development community around this game which a lot of my work is based on.

The main components of the game worth automating are:
- Movement
- Killing a target (enemy or player)
- Item management
- Equipment ehancements (alchemy)
  
## Timeline

#### Past
I've spent some time removing a lot of excess code and have trimmed down Drew Benton's loader and Weeman's phconnector proxy. I combined them into a single object to allow multiple bots in the same program to be a straightforward expansion.

#### Present
Currently, the focus is on an extensible framework that abstracts the packet parsing process and adds an internal event layer for information transmission.

#### Future
Near term, I'm starting to work on a pvp bot as a quick demoable proof of concept. The focus during this development will be good architecture. I plan to try to reuse as much code as possible for a grinding bot.
Long term, I want to create a 90ish cap chinese race-only, cooperating, and coordinating goldbot farm.
Ultralong term, the goal is to create a bot, that when set free in an empty server, can reach the level cap in the minimum amount of time and produce the strongest character possible. This includes but is not limited to:
- Advanced algorithms to select a dynamic leveling route up to the level cap
- A main dynamic party optimized for killing monsters in the specific leveling spot
- Additional luring bots to feed monsters to the main party
- Bot farms in the fields to collect gold, resources, and equipment for the main party
- Bot assembly lines to filter and enhance the best equipment for the main party
- An academy farm running to give King/Gold/Silver/Bronze Honor Buffs to the main character and party
- Dungeon running parties to build Egyptian sets and find Egyptian weapons & shields for the main character and party
- Pvp bots battling in battle arena for coins to build Egyptian accessories for the main character and party

## Other bots as inspiration
There are a few popular single-character bots that have been used by players such as:
- mBot
- Sbot
- Centerbot
- phBot

Some people have also made some tools like an automatic alchemy tool or an auto staller.

## Documentation
More of my documentation can be found in the [documentation directory](documents).

