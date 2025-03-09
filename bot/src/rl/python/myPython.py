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
def selectAction(model, observation, key):
  # jax.debug.print('Picking action for obs {}', observation)
  values = model(observation)
  return jnp.argmax(values)