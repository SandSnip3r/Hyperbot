# UI Redesign Proposal - Version 3: Dashboard with Draggable Panels

**Status: Implemented**

## Overview
A customizable dashboard where each type of information is a draggable widget. Users can arrange the layout to suit different monitoring scenarios.

## Fleet View
- Default layout shows a table of all characters and a chart of overall metrics.
- Additional widgets can display aggregated statistics such as average HP or number of active duels.

## Detail View
- Dragging a character from the table opens a new widget showing their details.
- Each detail widget includes state machine hierarchy, cooldowns, item counts and RL preferences.
- Widgets can be rearranged or closed independently.

## Advantages
- Highly flexible and user configurable.
- Useful for large monitors or multi-screen setups.

## Drawbacks
- Requires more initial setup from the user.
- Might be overwhelming for new operators.
