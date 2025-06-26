# Plan for Migrating from DDQN to C51

**Author:** Google DeepMind (via Gemini)
**Date:** June 26, 2025
**Context:** This document outlines the engineering plan to transition the reinforcement learning agent from a Double DQN (DDQN) with Prioritized Experience Replay (PER) architecture to a Categorical DQN (C51) architecture, also with PER.

## 1. Motivation

The current DDQN agent models the expectation of future returns, which can be limiting in a highly stochastic environment like an MMORPG. Distributional RL, starting with C51, models the full distribution of returns, not just the mean. This provides a richer training signal, which has been shown to lead to more stable learning and better final performance, especially in environments with complex reward structures.

This migration serves two primary goals:
1.  **Learning:** To gain hands-on experience with distributional reinforcement learning algorithms.
2.  **Performance:** To build a more robust and higher-performing PVP agent.

## 2. High-Level Plan

The migration will be broken down into three main phases, focusing on modifying the Python backend first, then the C++/Python interface, and finally the C++ training loop.

1.  **Phase 1: Python (JAX) Model & Algorithm Update**
    *   Modify the neural network to output a probability distribution for each action.
    *   Implement the C51 Bellman update and loss function.
2.  **Phase 2: C++/Python Interface (`JaxInterface`) Update**
    *   Adjust the data passed to and returned from the core `train` function.
3.  **Phase 3: C++ Training Manager (`TrainingManager`) Update**
    *   Update the PER priority calculation to use the new loss.
    *   Adjust logging and metrics for the new distributional paradigm.

---

## 3. Detailed Implementation Steps

### Phase 1: Python (`bot/src/rl/python/dqn.py`)

The core logic of the new algorithm resides here. It would be best to create a new file, `c51.py`, to keep the concerns separate, and have `JaxInterface` select which module to load based on configuration.

#### 1.1. Define C51 Hyperparameters in C++
To keep hyperparameters centralized, we will define the C51 parameters in `trainingManager.hpp` and pass them to the Python environment during initialization.

**In `bot/src/rl/trainingManager.hpp`:**
```cpp
// ... existing constants
static constexpr float kC51Vmin = -10.0f; // Minimum possible return, needs tuning
static constexpr float kC51Vmax = 10.0f; // Maximum possible return, needs tuning
static constexpr int kC51NumAtoms = 51;  // Standard for C51
```

**In `bot/src/rl/jaxInterface.cpp`:**
The `initialize` method should be updated to accept these values and pass them to the Python script.

**In `bot/src/rl/python/c51.py`:**
The Python code will then receive these values and construct the support.
```python
# In the Python script's initialization function
def initialize(v_min, v_max, num_atoms, ...):
    global V_MIN, V_MAX, NUM_ATOMS, SUPPORT, DELTA_Z
    V_MIN = v_min
    V_MAX = v_max
    NUM_ATOMS = num_atoms
    SUPPORT = jnp.linspace(V_MIN, V_MAX, NUM_ATOMS)
    DELTA_Z = (V_MAX - V_MIN) / (NUM_ATOMS - 1)
    # ... rest of initialization
```

#### 1.2. Modify Network Architecture
The output of the network needs to change from Q-values to a set of probabilities for each atom, for each action.

*   **Current (DDQN):** The final layer outputs a tensor of shape `(num_actions,)`.
*   **New (C51):** The final layer must output a tensor of shape `(num_actions, NUM_ATOMS)`. This tensor contains the logits for the distribution. We will apply a `softmax` operation across the `NUM_ATOMS` dimension to get the probability distribution for each action.

```python
# Example using Flax
# ... inside the model's __call__ method
x = nn.Dense(features=self.num_actions * NUM_ATOMS)(x)
logits = x.reshape((self.num_actions, NUM_ATOMS))
# The raw logits are returned; softmax is applied in the loss function or action selection
```

#### 1.3. Implement the C51 Loss Function & Bellman Update
This is the most critical change. The MSE loss will be replaced with a cross-entropy loss. The `train` function will need to perform the C51 projection.

**Steps for the `train` function:**
1.  Get the next-state distributions `p(s', :)` from the **target network** for all actions.
2.  Calculate the expected Q-values for the next state: `Q(s', a) = Σ_i (z_i * p_i(s', a))`.
3.  Select the best next action using the **online network's** Q-values (the Double DQN part): `a* = argmax_a Q_online(s', a)`.
4.  Use `a*` to select the corresponding next-state distribution from the **target network**: `p(s', a*)`.
5.  Compute the projected Bellman update for each atom `z_j` of the support:
    *   `Tz_j = rewards + (1 - terminals) * gamma * SUPPORT`
    *   Clip `Tz_j` to be within `[V_MIN, V_MAX]`.
6.  Calculate the indices and weights for projecting `Tz_j` onto the original `SUPPORT`.
    *   `b = (Tz_j - V_MIN) / DELTA_Z`
    *   `l = floor(b)`, `u = ceil(b)`
7.  Distribute the probability `p_j(s', a*)` to the neighboring atoms `l` and `u` in the target distribution `m`.
    *   `m_l += p_j(s', a*) * (u - b)`
    *   `m_u += p_j(s', a*) * (b - l)`
8.  The result, `m`, is a batch of target probability distributions.
9.  Calculate the cross-entropy loss between the online network's predicted distribution for the action taken `p(s, a)` and the target distribution `m`.
    *   `loss = -Σ_j (m_j * log(p_j(s, a)))`
10. Return this loss for each item in the batch. This will be the new priority for PER.

#### 1.4. Update Action Selection
The `select_action` function must be updated to work with distributions.
1.  Get the distributions for the current state from the online network: `p(s, :)`.
2.  Calculate the expected Q-value for each action: `Q(s, a) = Σ_i (z_i * p_i(s, a))`.
3.  Use these Q-values for the epsilon-greedy policy as before: `argmax_a Q(s, a)`.

### Phase 2: C++/Python Interface (`JaxInterface`)

The `JaxInterface` class acts as the bridge. The changes here are minimal and are mostly about data types.

#### 2.1. Update `TrainAuxOutput`
In `jaxInterface.hpp`, the `TrainAuxOutput` struct currently holds `tdErrors`. This should be changed to reflect that we are now getting back a loss value.

```cpp
// In jaxInterface.hpp
struct TrainAuxOutput {
  // ... other fields
  std::vector<float> losses; // Was tdErrors
};
```

The implementation in `jaxInterface.cpp` will need to be updated to parse this new field name from the Python dictionary returned by the `train` function.

### Phase 3: C++ Training Manager (`trainingManager.cpp`)

This is the final step, connecting the new algorithm to our existing training infrastructure.

#### 3.1. Update PER Priority Calculation
In `TrainingManager::train`, the logic for updating the replay buffer priorities needs to use the new `losses` vector.

```cpp
// In TrainingManager::train()
// ...
const JaxInterface::TrainAuxOutput trainOutput = jaxInterface_.train(...);

// ...
// Update priorities
for (int i=0; i<sampleResult.size(); ++i) {
  // ...
  // The absolute value of the cross-entropy loss is a good priority.
  newPriorities.push_back(std::abs(trainOutput.losses.at(i)));
}
replayBuffer_.updatePriorities(ids, newPriorities);
// ...
```

#### 3.2. Update Logging
The metrics sent to TensorBoard should be updated for clarity.

```cpp
// In TrainingManager::train()
// ...
const float minLoss = *std::min_element(trainOutput.losses.begin(), trainOutput.losses.end());
const float meanLoss = std::accumulate(trainOutput.losses.begin(), trainOutput.losses.end(), 0.0f) / trainOutput.losses.size();
const float maxLoss = *std::max_element(trainOutput.losses.begin(), trainOutput.losses.end());
jaxInterface_.addScalar("Loss/Min", minLoss, trainStepCount_);
jaxInterface_.addScalar("Loss/Mean", meanLoss, trainStepCount_);
jaxInterface_.addScalar("Loss/Max", maxLoss, trainStepCount_);

// The Q-Value metrics will now represent the *expected* Q-values, which is fine.
// The Python train function should be modified to return these expected values for logging.
jaxInterface_.addScalar("Expected_Q_Value/Min", trainOutput.meanMinQValue, trainStepCount_);
// ... etc
```

## 4. Conclusion & Next Steps

This plan provides a clear path from DDQN to C51. The most complex part is the implementation of the C51 Bellman update and projection in Python. I recommend starting there and ensuring it is numerically stable and correct before integrating it with the C++ code.

**Recommendation:**
1.  Create `bot/src/rl/python/c51.py` and implement the model, loss, and action selection logic.
2.  Write unit tests in Python to verify the projection logic.
3.  Update `JaxInterface` and `TrainingManager` as described above.
4.  Begin training and carefully monitor the new loss curves and agent performance. The `V_MIN` and `V_MAX` hyperparameters will likely require some tuning based on observed episode returns.

I am ready to assist with the implementation details when you are ready to proceed.
