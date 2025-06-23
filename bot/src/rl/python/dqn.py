from flax import nnx
import jax
import jax.numpy as jnp
import optax
import orbax
import orbax.checkpoint

class DqnModel(nnx.Module):
  def __init__(self, observationSize: int, stackSize: int, actionSpaceSize: int, dropoutRate: float, rngs: nnx.Rngs):
    key = rngs.params()
    self.featureExtractor1 = nnx.Linear(observationSize+1+actionSpaceSize, 128, rngs=rngs)
    self.dropout1 = nnx.Dropout(rate=dropoutRate)
    self.featureExtractor2 = nnx.Linear(128, 64, rngs=rngs)
    self.dropout2 = nnx.Dropout(rate=dropoutRate)
    self.featureExtractor3 = nnx.Linear(64, 32, rngs=rngs)
    self.dropout3 = nnx.Dropout(rate=dropoutRate)

    self.finalLinear1 = nnx.Linear(32*stackSize + 1*stackSize + observationSize, 1024, rngs=rngs)
    self.dropout4 = nnx.Dropout(rate=dropoutRate)
    self.finalLinear2 = nnx.Linear(1024, 256, rngs=rngs)
    self.dropout5 = nnx.Dropout(rate=dropoutRate)
    self.finalLinear3 = nnx.Linear(256, actionSpaceSize, rngs=rngs)

  # Input is:
  # - pastObservationStack:      (stackSize, observationSize)
  # - pastObservationTimestamps: (stackSize, 1)
  # - pastActions:               (stackSize, actionSpaceSize)
  # - pastMask:                  (stackSize, 1)
  # - currentObservation:        (observationSize)
  def __call__(self, pastObservationStack, pastObservationTimestamps, pastActions, pastMask, currentObservation, deterministic, rngKey=None):
    rngStreams = {}
    if not deterministic:
      if rngKey is None:
        raise ValueError("rngKey must be provided when deterministic is False")
      rngKeys = jax.random.split(rngKey, 5)
      rngStreams = {
        'dropout1': nnx.Rngs(dropout=rngKeys[0]),
        'dropout2': nnx.Rngs(dropout=rngKeys[1]),
        'dropout3': nnx.Rngs(dropout=rngKeys[2]),
        'dropout4': nnx.Rngs(dropout=rngKeys[3]),
        'dropout5': nnx.Rngs(dropout=rngKeys[4]),
      }

    # Run feature extractor from ~204->128->64->32
    pastDataStack = jnp.concat([pastObservationStack, pastObservationTimestamps, pastActions], axis=1)

    pastDataStack = self.featureExtractor1(pastDataStack)
    pastDataStack = jax.nn.relu(pastDataStack)
    pastDataStack = self.dropout1(pastDataStack, deterministic=deterministic, rngs=rngStreams.get('dropout1'))

    pastDataStack = self.featureExtractor2(pastDataStack)
    pastDataStack = jax.nn.relu(pastDataStack)
    pastDataStack = self.dropout2(pastDataStack, deterministic=deterministic, rngs=rngStreams.get('dropout2'))

    pastDataStack = self.featureExtractor3(pastDataStack)
    pastDataStack = jax.nn.relu(pastDataStack)
    pastDataStack = self.dropout3(pastDataStack, deterministic=deterministic, rngs=rngStreams.get('dropout3'))

    # Flatten all data for final MLP
    flattenedPastData = pastDataStack.ravel()
    flattenedMask = pastMask.ravel()
    concattedData = jnp.concat([flattenedPastData, flattenedMask, currentObservation], axis=0)

    x = self.finalLinear1(concattedData)
    x = jax.nn.relu(x)
    x = self.dropout4(x, deterministic=deterministic, rngs=rngStreams.get('dropout4'))

    x = self.finalLinear2(x)
    x = jax.nn.relu(x)
    x = self.dropout5(x, deterministic=deterministic, rngs=rngStreams.get('dropout5'))

    x = self.finalLinear3(x)
    return x

def printWeights(model):
  _, params = nnx.split(model)
  print(params)

@nnx.jit
def selectAction(model, pastObservationStack, pastObservationTimestamps, pastActions, pastMask, currentObservation, actionMask):
  values = model(pastObservationStack, pastObservationTimestamps, pastActions, pastMask, currentObservation, deterministic=True)
  values += actionMask
  return jnp.argmax(values), values

# Function to compute weighted loss and TD error for a SINGLE transition
def computeWeightedLossAndTdErrorSingle(model, targetModel, transition, weight, gamma, rngKey):
  pastModelInputTuple, selectedAction, isTerminal, reward, currentModelInputTuple = transition

  # --- DDQN Target Calculation ---
  # Define the calculation for the non-terminal case using DDQN logic
  def ddqnTargetCalculation(_):
    # 1. Select the best action for currentModelInputTuple using the *main* model's Q-values.
    #    We don't need gradients for this action selection step itself.
    qValuesNextStateMain = model(*currentModelInputTuple, deterministic=True)
    bestNextAction = jnp.argmax(qValuesNextStateMain)

    # 2. Evaluate the Q-value of that *selected* action using the *target* model.
    qValuesNextStateTarget = targetModel(*currentModelInputTuple, deterministic=True)
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
  values = model(*pastModelInputTuple, deterministic=False, rngKey=rngKey)
  pred = values[selectedAction]

  # Calculate Huber loss (unweighted)
  unweightedLoss = optax.losses.huber_loss(pred, targetValue)

  # Calculate TD Error (unweighted)
  tdError = targetValue - pred

  # Apply Importance Sampling weight to the loss
  weightedLoss = weight * unweightedLoss

  # Return additional debugging information
  return weightedLoss, (tdError, jnp.min(values), jnp.mean(values), jnp.max(values))

def computeWeightedLossAndTdErrorBatch(model, targetModel, transitions, weights, gamma, rngKey):
  batched = jax.vmap(computeWeightedLossAndTdErrorSingle, in_axes=( None, None, (0, 0, 0, 0, 0), 0, None, None), out_axes=(0, 0))
  weightedLosses, (tdErrors, minValues, meanValues, maxValues) = batched(model, targetModel, transitions, weights, gamma, rngKey)
  return jnp.mean(weightedLosses), (tdErrors, jnp.min(minValues), jnp.mean(meanValues), jnp.max(maxValues))

@nnx.jit
def jittedTrain(model, optimizerState, targetModel, pastModelInputTuple, selectedActions, isTerminals, rewards, currentModelInputTuple, weights, gamma, rngKey):
  (meanLoss, auxOutput), gradients = nnx.value_and_grad(computeWeightedLossAndTdErrorBatch, has_aux=True)(model, targetModel, (pastModelInputTuple, selectedActions, isTerminals, rewards, currentModelInputTuple), weights, gamma, rngKey)
  globalNorm = optax.global_norm(gradients)
  optimizerState.update(gradients)
  return (globalNorm, *auxOutput)

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