import math
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# Function to calculate standard deviation for a binomial distribution
def binomial_std_dev(n, p):
  return np.sqrt(n * p * (1 - p))

def parse_data(file_name):
  with open(file_name, 'r') as file:
    lines = file.readlines()

  successes = [0 for x in range(0,15)]
  failures = [0 for x in range(0,15)]

  for line in lines:
    trial, result = map(str.strip, line.split())
    if result.lower() == 'success':
      successes[int(trial)] += 1
    elif result.lower() == 'fail':
      failures[int(trial)] += 1

  return successes, failures

def to_percentage(y, _):
  return f'{y * 100:.1f}%'

def wilson_score_interval(successes, total_trials, confidence=0.95):
  p = successes / total_trials
  z = 1.96  # Z-score for 95% confidence interval

  numerator_left = p + (z**2) / (2 * total_trials) - z * math.sqrt((p * (1 - p) / total_trials) + (z**2) / (4 * (total_trials**2)))
  numerator_right = p + (z**2) / (2 * total_trials) + z * math.sqrt((p * (1 - p) / total_trials) + (z**2) / (4 * (total_trials**2)))
  denominator = 1 + (z**2) / total_trials

  lower_bound = numerator_left / denominator
  upper_bound = numerator_right / denominator

  return lower_bound, upper_bound

def main():
  # Check if the file name is provided as a command-line argument
  if len(sys.argv) != 2:
    print("Usage: python script.py <file_name>")
    sys.exit(1)

  # Read data from the file
  file_name = sys.argv[1]
  successes, failures = parse_data(file_name)
  print(successes)
  print(failures)
  
  probability = []
  lower = []
  upper = []
  for i in range(0, len(successes)):
    trial_count = successes[i] + failures[i]
    probability.append(successes[i] / max(1,trial_count))
    if trial_count > 0:
      lower_local, upper_local = wilson_score_interval(successes[i], trial_count)
      print('{}: {} suc, {} fail, {}% prob. {} lower and {} upper'.format(i, successes[i], failures[i], probability[i]*100, lower_local, upper_local))
      print('{}, {}'.format(probability[i]-lower_local, upper_local-probability[i]))
      lower.append(lower_local)
      upper.append(upper_local)
    else:
      lower.append(probability[i])
      upper.append(probability[i])

  probability = probability[1:]
  lower = lower[1:]
  upper = upper[1:]
  elixirOnly = [.5,.4,.3,.19,.17,.17,.17,.17,.17,.12,.12,.12,.12,.12]
  elixirAndPowder = [1,.7,.5,.27,.25,.25,.25,.25,.25,.2,.2,.2,.2,.2]
  theory = [min(1, x+.05) for x in elixirAndPowder]

  plt.figure(figsize=(24,16))

  # Scatter plot
  trial_numbers = range(1, 15)  # Trials 0-14
  plt.scatter(trial_numbers, elixirOnly, color='blue', label='Elixir Only', s=10)
  plt.scatter(trial_numbers, elixirAndPowder, color='green', label='Elixir & Lucky Powder', s=10)
  plt.scatter(trial_numbers, probability, color='red', label='Elixir, Lucky Powder, & Luck Stone', s=10)
  plt.scatter(trial_numbers, theory, color='black', label='Luck Stone Theory; +5% additive', s=10)
  plt.fill_between(trial_numbers, lower, upper, color='red', alpha=0.1, label='95% CI')

   # Customize the plot
  plt.title('Alchemy Enhancement Data')
  plt.xlabel('Attempted +')
  plt.ylabel('Probability of success')
  plt.legend()
  plt.xticks(trial_numbers)
  plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percentage))
  plt.grid(True)

  plt.savefig('output_plot.png')

  # Show the plot
  plt.show()

if __name__ == "__main__":
  main()