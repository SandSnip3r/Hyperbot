What common code should exist between bot and UI?

UI needs to display

Character name
Current level
Experience (current / max)
Sp (current SP count, sp_exp current, and sp_exp max)

Character health (current / max)
Character mp (current / max)
Current active skills & debuffs?

Location (x,y and region name)

Inventory gold
Inventory items

Storage gold
Storage items

# Things that require pk2

## Minimap graphics
Display the minimap graphic as our map

## Navmesh
Would we want to display the obstacles on the minimap? That's a lot of rendering effort and might be counterintuitive for the user

## Item stuff
### Item icons
Display inventory and storage in a similar grid UI to the game (with icons)
Also display item icons when interfacing with the item pick filter
### Item data
Given a refId, what's the name of the item, max durability, etc.

## Skill stuff
### Skill icons
Display skill icons when choosing buffs/attacks
Also maybe displaying skill/buff icons if we want to show player active buffs/debuffs
### Skill data
Are we going to want to know how much SP is required to level up a skill?