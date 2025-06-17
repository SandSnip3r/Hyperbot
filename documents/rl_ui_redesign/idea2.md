# RL UI Redesign Idea 2: Card Grid with Live Thumbnails

## Overview
- Replace the table with a responsive grid of "character cards".
- Each card displays a small live thumbnail of the game client, HP/MP bars, and key state text.
- Cards are grouped by pair via colored borders or labels.
- Selecting a card opens a side pane with the detailed character view.

## Key Features
- **Visual thumbnails** quickly reveal if a character is stuck or idling.
- **Side pane** keeps the main grid visible while inspecting individual characters.
- Detail pane shows the state machine tree, item counts/cooldowns, and RL preferences.
- Filtering controls allow searching by name or state.

## Benefits
- Highly visual overview ideal for large fleets.
- Side pane workflow makes it fast to jump between characters without extra windows.
- Thumbnails help diagnose stuck clients at a glance.

## Status
Implemented in the `rl_ui` application.
