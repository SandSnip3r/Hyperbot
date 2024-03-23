import functools
import matplotlib.pyplot as plt
import sys
import numpy as np

print(sys.setrecursionlimit(5000))

# for goal in range(3, 7):
#     y = [calculate(0, elixirCount, goal) for elixirCount in x]

#     # Create a line chart
#     plt.plot(x, y, label=f'Goal +{goal}')

#     threshold = 0.9
#     for i, y_value in enumerate(y):
#         if y_value > threshold:
#             plt.annotate(f'{i+1} elixirs (+{goal})', xy=(x[i], y_value), xytext=(x[i] + 0.2, y_value + 0.1 + 0.05 * goal),
#                          arrowprops=dict(facecolor='black', arrowstyle='->'),
#                          bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
#             break

# # Add labels and a title
# plt.xlabel('Number of elixirs')
# plt.ylabel('Probability of achieving goal')
# plt.title('Probabilities starting from +0')

# # Add a legend
# plt.legend()

# # Display the plot
# plt.show()


probabilities = [ 1.0, 0.7, 0.5, 0.27, 0.25, 0.25, 0.25, 0.25, 0.25 ]

@functools.lru_cache(maxsize=None)
def takeAction(current, numElixirs, R):
  return probabilities[current] + (1-probabilities[current])*probability(0,numElixirs-1,current, R) >= R

@functools.lru_cache(maxsize=None)
def probability(current, numElixirs, goal, R):
  if current == goal:
    return 1.0
  if current + numElixirs < goal:
    return 0.0
  if not takeAction(current, numElixirs, R):
    return 0.0
  return probabilities[current]  * probability(current+1, numElixirs-1, goal, R) + \
    (1 - probabilities[current]) * probability(0,         numElixirs-1, goal, R)

# Define the colors for the gradient
start_color = np.array([255, 0, 0])  # Red
end_color = np.array([0, 255, 0])    # Blue

# Create a custom colormap from start_color to end_color
num_colors = 100
colors = np.zeros((num_colors, 3))
for i in range(3):
    colors[:, i] = np.linspace(start_color[i], end_color[i], num_colors) / 255.0
custom_cmap = plt.cm.colors.ListedColormap(colors)

maxGoal = 7
x = range(1,maxGoal+1)
R_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for i, R in enumerate(R_values):
  y = []
  for goal in range(1,maxGoal+1):
    current = goal-1
    count = 0
    while True:
      result = probability(current, count, goal, R)
      if result > 0.0:
        print(f'{count} minimum elixirs required for {goal-1}->{goal} with R={R}')
        # print(f'Probability of achieving goal is {result*100}%.')
        # print(f'Probability of getting back to +{current} is {probability(0, count-1, current)*100}%.')
        break
      count += 1
    y.append(count)
  plt.plot(x, y, color=custom_cmap(i / len(R_values)), label=f'R={R}')


# Add labels and a title
# plt.yscale('log')
plt.xlabel('Goal +')
plt.ylabel('Elixirs')
plt.title('Minimum Elixirs To Take Action')
plt.grid(True, which='both', axis='y')
plt.grid(True, which='both', axis='x')
plt.gca().yaxis.set_ticks_position('both')
# plt.gca().yaxis.set_label_position('left')
plt.tick_params(axis='y', direction='inout', labelright=True)

# Add a legend
plt.legend()

# Display the plot
plt.show()
  