from flax import nnx
import jax
import jax.numpy as jnp

class MyModel(nnx.Module):
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

def func(number):
  print(f'Hyperbot\'s first call into Python!!! Passed argument: {number}')

@nnx.jit
def selectAction(model, observation, actionMask, key):
  # jax.debug.print('Picking action for obs {} with action mask {}', observation, actionMask)
  values = model(observation)
  values += actionMask
  return jnp.argmax(values)

def train(model, optimizerState, olderObservation, selectedAction, reward, newerObservation):
  # Move model(oldObservation)[selectedAction] towards max_action(model(newObservation)) + reward
  def lossFunction(model, observation, actionIndex, target):
    values = model(observation)
    return jnp.mean(jnp.square(values[actionIndex] - target))

  originalValue = model(olderObservation)[selectedAction]
  targetValue = reward + jnp.max(model(newerObservation))
  gradients = nnx.grad(lossFunction)(model, olderObservation, selectedAction, targetValue)
  optimizerState.update(gradients)
  newValue = model(olderObservation)[selectedAction]
  jax.debug.print('Old value: {}, New value: {}, Target value: {}', originalValue, newValue, targetValue)