# RL UI Redesign Idea 1: Dashboard with Expandable Rows

## Overview
- Keep the existing table layout but group characters by pair.
- Each row shows a pair summary (names, HP bars, MP bars, quick state overview).
- Clicking a row expands to reveal both characters individually.
- Double‑clicking opens a dedicated detail dialog with state machine tree, skill cooldowns, and RL preferences.

## Key Features
- **Pair grouping** helps reduce clutter when controlling many characters.
- **Expandable rows** allow quick inspection without losing context.
- In the detail dialog, show:
  - Full state machine hierarchy displayed as an indented tree.
  - Skill cooldown list with icons (existing).
  - Item counts and cooldowns for HP/MP potions and universal pills.
  - Table of RL action preferences with current probability or value.

## Benefits
- Familiar table format for quick scanning.
- Easy to collapse/expand pairs based on operator interest.
- Uses existing dialog concept so minimal code changes to integrate.
