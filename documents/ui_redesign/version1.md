# UI Redesign Proposal - Version 1: Split Panel Overview

## Overview
This design focuses on a split panel layout. The left side shows the fleet overview while the right side displays detailed information for selected characters.

## Fleet View
- A scrollable table lists all characters with columns for HP, MP, and current state machine.
- Selecting a row opens the detail view in the right panel instead of a dialog.

## Detail View
- Shows HP/MP bars, cooldown timers for skills and key items (HP/MP potions, universal pills).
- Displays full state machine hierarchy as a tree.
- Presents the RL agent's preferences for the top actions, updating in real time.

## Advantages
- Immediate context switching between fleet overview and details.
- Easier to monitor many characters while keeping a few details visible.

## Drawbacks
- Limited screen real estate when many characters are selected.
