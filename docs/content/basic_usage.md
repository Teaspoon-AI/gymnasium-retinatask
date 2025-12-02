# Basic Usage

This page covers the basic usage patterns for the Retina Task environment.

## Creating an Environment

The simplest way to create the environment:

```python
import gymnasium as gym
import gymnasium_retinatask

env = gym.make("RetinaTask-v0")
```

## Single Episode

Run a complete episode:

```python
import gymnasium as gym
import numpy as np
import gymnasium_retinatask

env = gym.make("RetinaTask-v0")

# Reset to start episode
obs, info = env.reset(seed=42)
print(f"Pattern: {info['pattern']}")
print(f"Observation shape: {obs.shape}")

# Take an action (random classification)
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

print(f"Reward: {reward:.2f}")
print(f"Terminated: {terminated}")

env.close()
```

## Multiple Episodes

Training loop example:

```python
import gymnasium as gym
import gymnasium_retinatask

env = gym.make("RetinaTask-v0", mode="batch", batch_size=100)

num_episodes = 10
for episode in range(num_episodes):
    obs, info = env.reset()
    episode_reward = 0

    while True:
        # Your policy/agent here
        action = your_policy(obs)

        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        if terminated or truncated:
            break

    print(f"Episode {episode}: Reward = {episode_reward:.2f}")

env.close()
```

## Evaluation Modes

### Single Pattern Mode

Evaluate on one random pattern per episode (default):

```python
env = gym.make("RetinaTask-v0", mode="single_pattern")
obs, info = env.reset()
# Episode contains exactly 1 pattern
```

### Batch Mode

Evaluate on a fixed number of random patterns:

```python
env = gym.make("RetinaTask-v0", mode="batch", batch_size=100)
obs, info = env.reset()
# Episode contains 100 random patterns
```

### Full Evaluation Mode

Evaluate on all 256 possible patterns:

```python
env = gym.make("RetinaTask-v0", mode="full_evaluation")
obs, info = env.reset()
# Episode contains all 256 patterns in order
```

## Reward Types

### Paper Reward (Default)

Uses the fitness function from the original paper:

```python
env = gym.make("RetinaTask-v0", reward_type="paper")
# Reward = 1000.0 / (1.0 + total_error)
# Perfect classification: reward = 1000.0
# Random agent: reward ≈ 10
```

### Simple Reward

Returns negative error directly:

```python
env = gym.make("RetinaTask-v0", reward_type="simple")
# Reward = -error
# Perfect classification: reward = 0.0
# Worse is more negative
```

## Understanding Observations

Observations are 8-dimensional vectors representing the retina pixels:

```python
obs, info = env.reset()
print(obs)  # e.g., [1. 0. 1. 1. 0. 1. 0. 1.]

# Reshape to see left/right structure
left_pixels = obs[:4]   # First 4 pixels
right_pixels = obs[4:]  # Last 4 pixels
```

## Understanding Actions

Actions are 2-dimensional classification outputs:

```python
# Perfect classification example
from gymnasium_retinatask import RetinaPatterns

obs, info = env.reset()
pattern = info["pattern"]

# Get correct labels
left_label, right_label = RetinaPatterns.get_labels(pattern)

# Create perfect action
action = np.array([left_label, right_label], dtype=np.float32)
obs, reward, terminated, truncated, info = env.step(action)

# With perfect action, reward should be 1000.0
print(f"Reward: {reward:.2f}")
```

## Accessing Pattern Information

The environment provides useful information in the `info` dictionary:

```python
obs, info = env.reset()

# Available in reset() info
print(f"Current pattern: {info['pattern']}")  # 0-255
print(f"Total patterns: {info['total_patterns']}")

# Available in step() info after taking action
obs, reward, terminated, truncated, info = env.step(action)

print(f"Pattern error: {info['pattern_error']}")  # Error for this pattern
print(f"Total error: {info['total_error']}")  # Cumulative error
print(f"Patterns evaluated: {info['patterns_evaluated']}")  # Count
print(f"Correct left: {info['correct_left']}")  # Ground truth
print(f"Correct right: {info['correct_right']}")  # Ground truth
```

## Random Agent Example

A complete example with a random agent:

```python
import gymnasium as gym
import numpy as np
import gymnasium_retinatask

def run_random_agent(num_episodes=5):
    env = gym.make("RetinaTask-v0", mode="batch", batch_size=100)

    total_rewards = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0

        while True:
            # Random classification
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                break

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

    env.close()

    print(f"\nAverage reward: {np.mean(total_rewards):.2f}")
    print(f"Std deviation: {np.std(total_rewards):.2f}")

if __name__ == "__main__":
    run_random_agent()
```

## Next Steps

- Learn about [pattern analysis](pattern_analysis.md) to understand the task better
- See [tutorials](../tutorials/simple_agent.md) for more advanced examples
- Check the [API reference](../api/environment.md) for detailed documentation
