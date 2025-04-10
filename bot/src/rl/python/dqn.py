from flax import nnx
import jax
import jax.numpy as jnp
import optax
import orbax
import orbax.checkpoint

class DqnModel(nnx.Module):
  def __init__(self, inSize: int, outSize: int, rngs: nnx.Rngs):
    intermediateSize = 512
    key = rngs.params()
    self.linear1 = nnx.Linear(inSize, intermediateSize, rngs=rngs)
    self.linear2 = nnx.Linear(intermediateSize, outSize, rngs=rngs)

  def __call__(self, x):
    x = self.linear1(x)
    x = jax.nn.relu(x)
    x = self.linear2(x)
    return x

def printWeights(model):
  _, params = nnx.split(model)
  print(params)

@nnx.jit
def selectAction(model, observation, actionMask, key):
  # jax.debug.print('Picking action for obs {} with action mask {}', observation, actionMask)
  values = model(observation)
  values += actionMask
  return jnp.argmax(values)

# Function to compute weighted loss and TD error for a SINGLE transition
def computeWeightedLossAndTdError(model, targetModel, transition, weight):
  observation, selectedAction, isTerminal, reward, nextObservation = transition

  # --- DDQN Target Calculation ---
  # Define the calculation for the non-terminal case using DDQN logic
  def ddqnTargetCalculation(_):
    # 1. Select the best action for nextObservation using the *main* model's Q-values.
    #    We don't need gradients for this action selection step itself.
    qValuesNextStateMain = model(nextObservation)
    bestNextAction = jnp.argmax(qValuesNextStateMain)

    # 2. Evaluate the Q-value of that *selected* action using the *target* model.
    qValuesNextStateTarget = targetModel(nextObservation)
    # Index the target Q-values with the action chosen by the main model
    targetQValueForBestAction = qValuesNextStateTarget[bestNextAction]

    # 3. Calculate the final target value (reward + discounted future value).
    #    Stop gradients from flowing back through the target network calculation.
    return reward + gamma * jax.lax.stop_gradient(targetQValueForBestAction)

  gamma = 1.0
  targetValue = jax.lax.cond(
      isTerminal,
      lambda _: reward,
      ddqnTargetCalculation,
      None
  )

  # Prediction from main model
  values = model(observation)
  pred = values[selectedAction]

  # Calculate Huber loss (unweighted)
  unweightedLoss = optax.losses.huber_loss(pred, targetValue)

  # Calculate TD Error (unweighted)
  tdError = targetValue - pred

  # Apply Importance Sampling weight to the loss
  weightedLoss = weight * unweightedLoss

  return weightedLoss, tdError

@nnx.jit
def train(model, optimizerState, targetModel, observation, selectedAction, isTerminal, reward, nextObservation, weight):
  (loss, tdError), gradients = nnx.value_and_grad(computeWeightedLossAndTdError, has_aux=True)(model, targetModel, (observation, selectedAction, isTerminal, reward, nextObservation), weight)
  optimizerState.update(gradients)
  return tdError

def getCopyOfModel(model, targetNetwork):
  graph, params = nnx.split(model)
  targetGraph, targetParams = nnx.split(targetNetwork)
  return nnx.merge(targetGraph, params)

def checkpointModel(model, path):
  print(f'Checkpointing model to {path}')
  graph, params = nnx.split(model)
  with orbax.checkpoint.StandardCheckpointer() as checkpointer:
    checkpointer.save(path, params)
    checkpointer.wait_until_finished()

def checkpointOptimizer(optimizer, path):
  print(f'Checkpointing optimizer to {path}')
  with orbax.checkpoint.StandardCheckpointer() as checkpointer:
    checkpointer.save(path, nnx.state(optimizer))
    checkpointer.wait_until_finished()

def loadModelCheckpoint(currentModel, path):
  print(f'Loading model checkpoint from {path}')
  graph, abstractParams = nnx.split(currentModel)
  with orbax.checkpoint.StandardCheckpointer() as checkpointer:
    loadedParams = checkpointer.restore(path, abstractParams)
    return nnx.merge(graph, loadedParams)

def loadOptimizerCheckpoint(optimizer, path):
  print(f'Loading optimizer checkpoint from {path}')
  with orbax.checkpoint.StandardCheckpointer() as checkpointer:
    abstractOptStateTree = jax.tree_util.tree_map(orbax.checkpoint.utils.to_shape_dtype_struct, nnx.state(optimizer))
    optimizerState = checkpointer.restore(path, abstractOptStateTree)
    nnx.update(optimizer, optimizerState)
    return optimizer