# Hyperbot
*A Multi-Agent Reinforcement Learning Framework for Silkroad Online*

[![Watch the Demo](https://img.youtube.com/vi/6NusfkLRzpk/0.jpg)](https://youtube.com/watch?v=6NusfkLRzpk)

---

## Current Status
**Status:** Experimental

Hyperbot is an actively developed research framework designed to connect reinforcement learning with the rich, dynamic world of *Silkroad Online* — a live MMORPG featuring complex combat systems, player interactions, inventories, trading, and long-term progression.

The project currently supports **1v1 Player-vs-Player (PvP)** battles through a fully automated agent interface. Multiple characters can be controlled concurrently, allowing large-scale data collection in real-time. Agents interact directly with the game server through a low-level TCP interface — no game client is required.

While the current API is not yet finalized, all necessary infrastructure exists for defining and training RL agents in **real-time asynchronous environments**, where the world continues evolving regardless of inference latency.

---

## Overview
Hyperbot is an ambitious exploration into **Reinforcement Learning in a Live Online Multiplayer World**. Unlike synthetic or simulated benchmarks, Hyperbot operates inside an authentic MMORPG, making it a unique testbed for developing agents that must reason, plan, and adapt in persistent, human-designed environments.

The framework’s long-term goal is to enable agents that can **understand and master MMORPG gameplay** — from individual duels to coordinated multi-agent combat, cooperative monster hunting, dynamic economies, and large-scale strategy.

---

## Why an MMORPG?
Most reinforcement learning environments today are either:
- **Highly abstract** (e.g. grid worlds, Atari), or
- **Fully simulated** and **resettable** (e.g. Mujoco, Crafter, Procgen).

By contrast, *Silkroad Online* offers:
- **Persistent State** – The world evolves even when the agent does not act.
- **Rich Semantics** – Combat, trading, exploration, skill trees, inventory systems, and social dynamics.
- **Hierarchical Goals** – From micro-level tactics (skill & item use, positioning) to macro-level strategy (gear optimization, party formation).
- **Partial Observability** – Information is incomplete, noisy, and temporally extended.
- **Multi-Agent Interactions** – Both cooperative and adversarial behaviors emerge naturally.

These properties make MMORPGs a fertile environment for *next-generation RL research* — where agents must operate asynchronously, generalize across long horizons, and make decisions with delayed or sparse feedback.

---

## Project Goals

### Short-Term
- Develop and benchmark agents for **1v1 PvP combat**.
- Formalize the asynchronous API for agent interaction.
- Document baseline performance (non-acting, random, scripted, and a landmark RL algorithm).

### Medium-Term
- Support multiple sub-environments within the overall MMORPG.
- Explore curriculum learning for complex behaviors (e.g. group tactics).
- Introduce flexible reward shaping and evaluation metrics.

### Long-Term
- Build agents capable of **holistic MMORPG mastery**, including:
  - Dynamic questing and leveling routes
  - Resource management and equipment optimization
  - Multi-agent coordination in large-scale battles
  - Strategic reasoning across thousands of concurrent states

---

## Key Features
- **Real MMORPG Environment** – Interact with a production-grade game world featuring authentic network protocols and state transitions.
- **Asynchronous Control** – The environment evolves continuously; agents must act in real time.
- **Multi-Agent Capability** – Designed to manage and coordinate multiple characters concurrently.
