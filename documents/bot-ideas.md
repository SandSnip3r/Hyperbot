# Intelligence ideas

***
## Attacking players
##### Automated attacking players - same style as current mbot trick
- list of skills for enemies
  - maybe categories for enemies for prioritization
  - basically a byproduct of automated dumb buffing
##### Automated stationary pvp - the idea is a perfect fighter
- calculate defense of player
  - based on level, set, build, buffs, scrolls
- learn how much damage skills can do
- estimate HP/MP
- estimate recovery rate
- choose an attack that is most likely to kill
##### Automated mobile pvp - pretty close emulation of a human player in fortress war
- requires automated stationary pvp
- requires automated navigation
  - to run away from or chase enemies
  - to evade then return to strike?
  - aware of safe-zone

***
## Navigation
#### Automated scripts - follow a pre-defined path
- go to a location, cast a skill, kill all mobs, use an item, teleport, wait, zerk
#### Automated navigation - move to any desired point
- probably relatively straightforward graph algorithms
- avoid dangerous mobs on route
- run from enemy?
- inside the PK2, there are some files for navigation
- if this is difficult, collect data for navigatable map (not really scaleable, need to coordinate on back end)

***
## Buffs
#### Automated dumb buffing - basically what mbot already does
- mbot-like skill choices and character selection
#### Automated intelligent buffing - mbot + targeted buffs on certain conditions
- requires automated dumb buffing
- maybe not much more than mconditionals
- might not want to give the player all screens at once, maybe give phys/mag then ultimate after

***
## Attacking monsters
#### Automated dumb grinding - basically what mbot already does
- required automated townloop
- requires automated pickup
- requires automated dumb buffing
- requires automated scripts
  - for navigation and lure
- mbot-like skill choices
- configurable berzerk
  - maybe nobody zerked at the same time
  - maybe only zerk if someone else is zerked
#### Automated intelligent grinding - mbot + better monster attack choices, maybe targeted KSing
- requires automated intelligent buffing
- required automated townloop
- requires automated pickup
- requires automated navigation
- does the pk2 contain mob spawn locations?
- intelligent lure
  - get more enemies based on party's killing capacity
  - grab from large groups of enemies (or small if appropriate)
- avoid mobs' special attacks which give disabling status (sireness's and Beakyung's petrify, step to the side)
- ability to switch out gear for better gear that maybe came from a drop
  - or from another member's drop
    - requires B2B communication
- ability to auto-alchemy
- between logged bots (B2B)
  - request buffs
    - "need defense", "need heal", "knockback mobs around me"
  - request items
  - academy matching
  - if someone is going to town in a party, maybe members ask him to bring potions or something
  - share return scrolls (they can drop when you die and mbot will get stuck if a char drops his)
- intelligent mob selection
  - all 4 wizards shouldnt drop a meteor on a little general mob
  - kill the weakest first to minimize enemies
- return to town the way we came (maybe if no return scrolls)
#### Automated unique monster hunting - preemtively or responsively, go hunt unique monsters
- requires automated navigation
- requires automated dumb grinding
- will need to find out how to know unique spawnpoints
- probably dont want to go for titans
- zerk-run becomes a thing
#### Automated multi-location plvl - ability to plvl any character from 1 to near-cap
- requires B2B communication
- requires automated navigation
- requires automated dumb grinding
- requires automated leveling skills

***
## Other
#### Automated FGW - maximize FGW throughput
- requires automated dumb grinding
- requires automated scripts
  - or could use automated navigation
- keep request queue for dimension holes for every bot that wants to do one
  - requires B2B
  - if dimension hole picked, give to highest priority bot
- watch cooldown to re-enter as quickly as possible
- dont re-enter within certain time window if player wants to do one himself manually
- spawn and enter FGW

#### Automated job temple unique hunting - kill uniques for coins
- requires automated dumb grinding
- requires automated mobile pvp
- requires automated scripts
  - or could use automated navigation

#### Automated alchemy - plus, blues, whites
- requires automated townloop
- if working on multiple items, maybe add configurable priority or always improve worst first
- request resources from other bots
  - requires B2B
- dont alchemy unless its likely (configurable % probability?) that we can return to the same state

#### Automated trade runs - maximize trade run throughput
- requires automated mobile pvp
- requires automated navigation
- requires automated dumb grinding
- buy from NPC, sell to NPC
- maybe go to most profitable town

#### Automated pickup - pick items off the ground
- closest, least busy, and safest char should pick
  - requires B2B
  - maybe unsafe char runs through party to pick (to bring mob into party)
- should pick closest item
- should prioritize valuable items
  - will need to allow user to configure value of items
  - if someone dies and drops something, this should be pretty valuable

#### Automated academy - reach any honor rank
- form matching, create new chars, join academy, take to plvl spot, maybe move to new plvl spot, reach 40, graduate, delete
- requires automated multi-location plvl
- requires automated character deletion
- requires automated character creation

#### Automated townloop - restock on items, store items, maybe some alchemy
- requires automated storage sorting
- requires automated buying from NPC
- find quickest townloops
  - option to skip NPC if possible
  - maybe teleport to another town for a quicker townloop
- option to place items in guild storage
- look into a buyback option (current items are lost because of however mbot works)

#### Automated stalling - sell items in a stall
- maybe put prices for all items in inventory (to replenish empty stall slots when items are sold)
- maybe put prices for all items in storage even?
- cycling titles?
- titles based on items in stall?

#### Automated party matching - form and join parties in matching
- can be required for multi-char FGW runs
- can be useful for advertising

#### Automated quests - get, complete, turn in quests
- could be done with quest scripts
  - requires automated scripts

#### Automated leveling skills - as the character levels up, or skillpoints are gained, level up the char's skills
- configure unusued SP (amount to keep in "bank")
- configurable prioritizition of masteries, trees, and skills

#### Automated adding stat points - as the character levels up, add stat points
- configurable ratio

#### Automated character creation - create a new character with given name, race, starting gear
- allow random names or following some pattern

#### Automated character deletion - delete an existing character with a given name on a given account

#### Automated inventory sorting - place things in stacks

#### Automated storage sorting - place things in stacks, maybe in a specific order

#### Automated guild storage sorting - place things in stacks, maybe in a specific order

#### Automated buying from NPC - given an item and an NPC, buy a quantity of an item

***
### Items
- add option to automatically upgrade for adv elixir
  - maybe do some alchemy (to get to +5 for chance at better adv elixir)

### Uncategorized
- need some way to kill mobs along a path or at least protect certain party members
  - for automated multi-location plvl
  - for automated FGW
- investigate anti-KS measures
  - lure extra to enemy to cause them to die
  - debuff
  - block buffs
  - drop items that force return-to-town
  - move away from KSing party
- consider commanding a bot on another computer
  - maybe add another bot username to trusted list for certain characters
  - for coordinated party with friends
- storage characters
    all characters have some storage thresholds set, when enough of the characters pass the threshold (like all characters have <12 storage slots left) then- they all decide to go to town, login storage characters, and do the trading
  - have a list of id, pass, items to store, free spots to leave
  - some way to request items
    - character will do alchemy, needs lots of items, register some storage characters as sources for items
- command line
  - some kind of bash interpreter to do things

### Data collection
- alchemy
  - compare success rates of different items
    - sro original website said that different items have different inherent lucks
      - if true and measureable, some "lucky" items should be worth much more
  - investigate success rate given a previous failure (old superstition)
  - investigate success rate given a relog (old superstition)
- unique monster kills & globals
  - maybe the server has a clientless that just idly monitors data
  - maybe new server owners create a character and give login info and maybe pk2 file for this clientless (cant automate all newly created pservers)

### Marketplace
- requires automated navigation
- allow automated buying and selling of items with other people with this tool
  - the bot will facilitate a safe trade
    - consider scammers who will want to kill client after 1/2 transaction
    - maybe a drop-trade? in a safe area?
    - or consignment
    - or a mediating character run by my server (logs in, takes both items, distributes, logs off)
- a webpage linked to bot account to "buy/sell"
  - similar to eBay

### Per-bot logging
- typical mbot logging
  - better filtered item drop log
- windows notifications on certain events
- really nice statistics section
  - graphs for gold, drop, exp, sp rate
  - predicted time until town again
  - forgotten world rate
    - drop rate
  - quest statistics
    - compare exp/sp from mobs vs quests

### Bot-sum statistics
- show total wealth
- some total statistics
- rank characters

### Global statistics
- aggregate some things from all users

### Smartphone connection
- receive notifications on certain events
- view statistics
- stop, start, restart, change mode, kill

### Bot use ideas
- put a char at every entrance/exit of town to monitor traffic, specifically to protect automated trade runs
