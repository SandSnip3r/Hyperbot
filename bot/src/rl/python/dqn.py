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
    self.linear2 = nnx.Linear(intermediateSize, intermediateSize, rngs=rngs)
    self.linear3 = nnx.Linear(intermediateSize, outSize, rngs=rngs)

  def __call__(self, x):
    x = self.linear1(x)
    x = jax.nn.relu(x)
    x = self.linear2(x)
    x = jax.nn.relu(x)
    x = self.linear3(x)
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
def computeWeightedLossAndTdErrorSingle(model, targetModel, transition, weight, gamma):
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

  return weightedLoss, (tdError, jnp.min(values), jnp.mean(values), jnp.max(values))

def computeWeightedLossAndTdErrorBatch(model, targetModel, transitions, weights, gamma):
  batched = jax.vmap(computeWeightedLossAndTdErrorSingle, in_axes=( None, None, (0, 0, 0, 0, 0), 0, None ), out_axes=(0, 0))
  weightedLosses, (tdErrors, minValues, meanValues, maxValues) = batched(model, targetModel, transitions, weights, gamma)
  return jnp.mean(weightedLosses), (tdErrors, jnp.mean(minValues), jnp.mean(meanValues), jnp.mean(maxValues))

@nnx.jit
def jittedTrain(model, optimizerState, targetModel, observations, selectedActions, isTerminals, rewards, nextObservations, weights, gamma):
  npSelectedActions = jnp.array(selectedActions)
  npIsTerminals = jnp.array(isTerminals)
  npRewards = jnp.array(rewards)
  npWeights = jnp.array(weights)

  (meanLoss, auxOutput), gradients = nnx.value_and_grad(computeWeightedLossAndTdErrorBatch, has_aux=True)(model, targetModel, (observations, npSelectedActions, npIsTerminals, npRewards, nextObservations), npWeights, gamma)
  optimizerState.update(gradients)
  return auxOutput

def convertThenTrain(model, optimizerState, targetModel, observations, selectedActions, isTerminals, rewards, nextObservations, weights, gamma):
  # print(f'observations: {type(observations)}.{observations.shape}: {observations}')
  # print(f'selectedActions: {type(selectedActions)}: {selectedActions}')
  # print(f'isTerminals: {type(isTerminals)}: {isTerminals}')
  # print(f'rewards: {type(rewards)}: {rewards}')
  # print(f'nextObservations: {type(nextObservations)}.{nextObservations.shape}: {nextObservations}')
  # print(f'weights: {type(weights)}: {weights}')

  npSelectedActions = jnp.array(selectedActions)
  npIsTerminals = jnp.array(isTerminals)
  npRewards = jnp.array(rewards)
  npWeights = jnp.array(weights)

  result = jittedTrain(model, optimizerState, targetModel, observations, npSelectedActions, npIsTerminals, npRewards, nextObservations, npWeights, gamma)
  jax.block_until_ready(result)
  return result

def getCopyOfModel(model, targetNetwork):
  graph, params = nnx.split(model)
  targetGraph, targetParams = nnx.split(targetNetwork)
  return nnx.merge(targetGraph, params)

@nnx.jit
def polyakUpdateTargetModel(model, targetModel, tau):
  """
  Perform Polyak averaging (soft update) of model parameters to target model.

  targetParams = (1 - tau) * targetParams + tau * sourceParams

  Args:
      model: Source model
      targetModel: Target model to be updated
      tau: Interpolation parameter (0 < tau < 1)

  Returns:
      Updated target model
  """
  # Extract parameters
  sourceGraph, sourceParams = nnx.split(model)
  targetGraph, targetParams = nnx.split(targetModel)

  # Perform the Polyak averaging
  def update_param(source_param, target_param):
    return (1 - tau) * target_param + tau * source_param

  # Apply the update to each parameter
  updated_params = jax.tree_util.tree_map(update_param, sourceParams, targetParams)

  # Merge back the graph with updated parameters
  return nnx.merge(targetGraph, updated_params)

def checkpointModel(model, path):
  print(f'Checkpointing model to {path}')
  graph, params = nnx.split(model)
  with orbax.checkpoint.StandardCheckpointer() as checkpointer:
    checkpointer.save(path, params, force=True)
    checkpointer.wait_until_finished()

def checkpointOptimizer(optimizer, path):
  print(f'Checkpointing optimizer to {path}')
  with orbax.checkpoint.StandardCheckpointer() as checkpointer:
    checkpointer.save(path, nnx.state(optimizer), force=True)
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

def getOptaxAdamWOptimizer(learningRate):
  # Do not apply weight decay to the biases
  def isKernel(x):
    return jax.tree.map(lambda y: y.ndim > 1, x)

  adamW = optax.adamw(
      learning_rate=learningRate,
      weight_decay=1e-2,
      mask=isKernel
  )

  tx = optax.chain(
    optax.clip_by_global_norm(1.0),
    adamW
  )
  return tx