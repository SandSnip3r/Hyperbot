import functools
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

probabilities = [1.0,0.7,0.5,0.27,0.25,0.25,0.25,0.25,0.25,0.20,0.20,0.20]

@functools.lru_cache(maxsize=None)
def calculate(current, numElixirs, goal):
  if current == goal:
    return 1.0
  if numElixirs == 0:
    return 0.0
  return probabilities[current+1] * calculate(current+1, numElixirs-1, goal) + \
    (1 - probabilities[current+1]) * calculate(0, numElixirs-1, goal)

def plot1():
  def percentage_formatter(x, pos):
    return f'{x*100:.0f}%'
  
  plt.gca().yaxis.set_major_formatter(FuncFormatter(percentage_formatter))

  x = range(0,4000)

  for goal in range(2,7):
    y = [calculate(0,elixirCount,goal) for elixirCount in x]

    # Create a line chart
    plt.plot(x, y, label=f'Goal +{goal}')

    threshold = 0.9
    for i, y_value in enumerate(y):
      if y_value > threshold:
        # plt.text(x[i] + 0.2, y_value + 0.1 - 0.05 * goal, f'{i+1} elixirs (+{goal})', rotation=45, ha='left', va='center', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
        # plt.annotate(f'Point {i+1}', xy=(x[i], y_value), xytext=(x[i] + 0.2, y_value + 0.1), arrowprops=dict(facecolor='black', arrowstyle='->'))
        plt.annotate(f'{i} elixirs (+{goal})', xy=(x[i], y_value), xytext=(x[i] + 0.2, y_value + 0.1 - 0.05 * goal), arrowprops=dict(facecolor='black', arrowstyle='->'), bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
        break

    # Add labels and a title
    plt.xlabel('Number of elixirs')
    plt.ylabel('Probability of achieving goal')
    plt.title('Probabilities starting from +0')

  # Add a legend
  plt.legend()

  # Display the plot
  plt.show()

def plot2():
  def percentage_formatter(x, pos):
    return f'{x*100:.0f}%'
  
  # plt.gca().yaxis.set_major_formatter(FuncFormatter(percentage_formatter))

  goals = [x for x in range(1,11)]
  req = []
  lastElixirCount = 0
  threshold = 0.9
  for goal in goals:
    count = 0
    while calculate(0,count,goal) <= threshold:
      count += 1
    print(count)
    req.append(count)
  print(goals)
  print(req)

  # goals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  # req = [2, 10, 47, 205, 840, 3384, 13563, 54281, 271450, 1357303]

  plt.plot(goals, req)

  for i, (x_val, y_val) in enumerate(zip(goals, req)):
    plt.text(x_val, y_val, f'{y_val}', ha='right', va='bottom', fontsize=8, color='black')

  # Add labels and a title
  plt.xlabel('Goal +')
  plt.ylabel('Number of elixirs')
  plt.title('Number of elixirs required to reach goal with at least 90% probability')

  # Add a legend
  # plt.legend()

  plt.xticks(goals)

  # Display the plot
  plt.show()

def main():
  sys.setrecursionlimit(2000000000)  # Set a higher limit
  # plot1()
  # plot2()

  count = 0
  while True:
    probability = calculate(0,count,5)
    if probability > 0.75:
      print(count)
      break
    count += 1

if __name__ == "__main__":
  main()