from flax import nnx
import jax
import jax.numpy as jnp
import optax
import orbax
import orbax.checkpoint

class DqnModel(nnx.Module):
  def __init__(self, observationSize: int, stackSize: int, actionSpaceSize: int, rngs: nnx.Rngs):
    key = rngs.params()
    self.featureExtractor1 = nnx.Linear(observationSize+1+actionSpaceSize, 128, rngs=rngs)
    self.featureExtractor2 = nnx.Linear(128, 64, rngs=rngs)
    self.featureExtractor3 = nnx.Linear(64, 32, rngs=rngs)

    self.finalLinear1 = nnx.Linear(32*stackSize + 1*stackSize + observationSize, 1024, rngs=rngs)
    self.finalLinear2 = nnx.Linear(1024, 256, rngs=rngs)
    self.finalLinear3 = nnx.Linear(256, actionSpaceSize, rngs=rngs)

  # Input is:
  # - pastObservationStack:      (stackSize, observationSize)
  # - pastObservationTimestamps: (stackSize, 1)
  # - pastActions:               (stackSize, actionSpaceSize)
  # - pastMask:                  (stackSize, 1)
  # - currentObservation:        (observationSize)
  def __call__(self, pastObservationStack, pastObservationTimestamps, pastActions, pastMask, currentObservation):
    # Run feature extractor from ~204->128->64->32
    pastDataStack = jnp.concat([pastObservationStack, pastObservationTimestamps, pastActions], axis=1)
    pastDataStack = self.featureExtractor1(pastDataStack)
    pastDataStack = jax.nn.relu(pastDataStack)
    pastDataStack = self.featureExtractor2(pastDataStack)
    pastDataStack = jax.nn.relu(pastDataStack)
    pastDataStack = self.featureExtractor3(pastDataStack)
    pastDataStack = jax.nn.relu(pastDataStack)

    # Flatten all data for final MLP
    flattenedPastData = jnp.concat(pastDataStack, axis=0)
    flattenedMask = jnp.concat(pastMask, axis=0)
    concattedData = jnp.concat([flattenedPastData, flattenedMask, currentObservation], axis=0)

    x = self.finalLinear1(concattedData)
    x = jax.nn.relu(x)
    x = self.finalLinear2(x)
    x = jax.nn.relu(x)
    x = self.finalLinear3(x)

    return x

def printWeights(model):
  _, params = nnx.split(model)
  print(params)

@nnx.jit
def selectAction(model, pastObservationStack, pastObservationTimestamps, pastActions, pastMask, currentObservation, actionMask, key):
  values = model(pastObservationStack, pastObservationTimestamps, pastActions, pastMask, currentObservation)
  values += actionMask
  return jnp.argmax(values)

# Function to compute weighted loss and TD error for a SINGLE transition
def computeWeightedLossAndTdErrorSingle(model, targetModel, transition, weight, gamma):
  pastModelInputTuple, selectedAction, isTerminal, reward, currentModelInputTuple = transition

  # --- DDQN Target Calculation ---
  # Define the calculation for the non-terminal case using DDQN logic
  def ddqnTargetCalculation(_):
    # 1. Select the best action for currentModelInputTuple using the *main* model's Q-values.
    #    We don't need gradients for this action selection step itself.
    qValuesNextStateMain = model(*currentModelInputTuple)
    bestNextAction = jnp.argmax(qValuesNextStateMain)

    # 2. Evaluate the Q-value of that *selected* action using the *target* model.
    qValuesNextStateTarget = targetModel(*currentModelInputTuple)
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
  values = model(*pastModelInputTuple)
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
def jittedTrain(model, optimizerState, targetModel, pastModelInputTuple, selectedActions, isTerminals, rewards, currentModelInputTuple, weights, gamma):
  (meanLoss, auxOutput), gradients = nnx.value_and_grad(computeWeightedLossAndTdErrorBatch, has_aux=True)(model, targetModel, (pastModelInputTuple, selectedActions, isTerminals, rewards, currentModelInputTuple), weights, gamma)
  optimizerState.update(gradients)
  return auxOutput

def convertThenTrain(model, optimizerState, targetModel, oldObservations, selectedActions, isTerminals, rewards, newObservations, weights, gamma):
  selectedActions = jnp.array(selectedActions)
  isTerminals = jnp.array(isTerminals)
  rewards = jnp.array(rewards)
  weights = jnp.array(weights)
  result = jittedTrain(model, optimizerState, targetModel, oldObservations, selectedActions, isTerminals, rewards, newObservations, weights, gamma)
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