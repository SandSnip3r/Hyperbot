from flax import nnx
import jax
import jax.numpy as jnp

class MyModel(nnx.Module):
  def __init__(self, inSize: int, outSize: int, *, rngs: nnx.Rngs):
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

def selectAction(observation, key):
  # Return a random number in the range [0,37]
  print(f'Given observation: "{observation}"')
  return jax.random.randint(key, (1,), 0, 37)[0]