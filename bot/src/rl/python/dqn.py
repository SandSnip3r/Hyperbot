from flax import nnx
import jax
import jax.numpy as jnp

class DqnModel(nnx.Module):
  def __init__(self, inSize: int, outSize: int, rngs: nnx.Rngs):
    intermediateSize = 64
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

@nnx.jit
def train(model, optimizerState, targetModel, olderObservation, selectedAction, isTerminal, reward, newerObservation):
  # Move model(oldObservation)[selectedAction] towards gamma * max_action(targetModel(newObservation)) + reward
  def lossFunction(model, observation, actionIndex, target):
    values = model(observation)
    return jnp.mean(jnp.square(values[actionIndex] - target))

  currentValue = model(olderObservation)[selectedAction]
  gamma = 0.9925
  targetValue = jax.lax.cond(isTerminal, lambda _: reward, lambda _: reward + gamma * jnp.max(targetModel(newerObservation)), None)

  gradients = nnx.grad(lossFunction)(model, olderObservation, selectedAction, targetValue)
  optimizerState.update(gradients)
  newValue = model(olderObservation)[selectedAction]
  # jax.debug.print('Old value: {}, New value: {}, Target value: {} Terminal? {}', currentValue, newValue, targetValue, isTerminal)

def getCopyOfModel(model, targetNetwork):
  graph, params = nnx.split(model)
  targetGraph, targetParams = nnx.split(targetNetwork)
  return nnx.merge(targetGraph, params)
