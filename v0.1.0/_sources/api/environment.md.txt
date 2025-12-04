# Environment API

## RetinaEnvV0

```{eval-rst}
.. autoclass:: gymnasium_retinatask.retina_env.RetinaEnvV0
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
```

## Environment Specifications

### Observation Space

```python
Box(0.0, 1.0, (8,), float32)
```

An 8-dimensional vector representing the retina pixels:
- Indices 0-3: Left side pixels (bits 7-4 of the pattern)
- Indices 4-7: Right side pixels (bits 3-0 of the pattern)
- Values: 0.0 (pixel off) or 1.0 (pixel on)

### Action Space

```python
Box(0.0, 1.0, (2,), float32)
```

A 2-dimensional vector for classification outputs:
- Index 0: Left side classification (0.0 = invalid, 1.0 = valid)
- Index 1: Right side classification (0.0 = invalid, 1.0 = valid)

### Parameters

#### mode

Episode evaluation mode:

- `"single_pattern"` (default): One random pattern per episode
- `"batch"`: Fixed number of random patterns per episode
- `"full_evaluation"`: All 256 patterns in sequence

#### batch_size

Number of patterns per episode when `mode="batch"`.

**Type:** `int`
**Default:** `100`
**Range:** `1` to `256`

#### reward_type

Reward function type:

- `"paper"` (default): Uses fitness function `1000.0 / (1.0 + error)`
- `"simple"`: Returns negative error directly

**Type:** `str`
**Options:** `"paper"`, `"simple"`

### Reset Info Dictionary

Returned by `env.reset()`:

| Key | Type | Description |
|-----|------|-------------|
| `pattern` | `int` | Current pattern (0-255) |
| `total_patterns` | `int` | Total patterns in episode |

### Step Info Dictionary

Returned by `env.step(action)`:

| Key | Type | Description |
|-----|------|-------------|
| `pattern` | `int` or `None` | Current pattern (None if terminated) |
| `correct_left` | `float` | Ground truth left label (0.0 or 1.0) |
| `correct_right` | `float` | Ground truth right label (0.0 or 1.0) |
| `pattern_error` | `float` | Error for current pattern |
| `total_error` | `float` | Cumulative error across episode |
| `patterns_evaluated` | `int` | Number of patterns evaluated |

### Termination

Episode terminates when all patterns in the episode have been evaluated.

- `terminated`: `True` when episode complete
- `truncated`: Always `False` (no time limits)

## Example Usage

### Basic Episode

```python
import gymnasium as gym
import gymnasium_retinatask

env = gym.make("RetinaTask-v0")
obs, info = env.reset(seed=42)

action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

print(f"Pattern: {info['pattern']}")
print(f"Error: {info['pattern_error']:.3f}")
print(f"Reward: {reward:.3f}")

env.close()
```

### Full Evaluation

```python
import gymnasium as gym
import numpy as np
from gymnasium_retinatask import RetinaPatterns

env = gym.make("RetinaTask-v0", mode="full_evaluation")
obs, info = env.reset()

total_correct = 0

while True:
    # Perfect classification
    pattern = info["pattern"]
    left, right = RetinaPatterns.get_labels(pattern)
    action = np.array([left, right], dtype=np.float32)

    obs, reward, terminated, truncated, info = env.step(action)

    if info["pattern_error"] == 0.0:
        total_correct += 1

    if terminated:
        break

accuracy = total_correct / info["patterns_evaluated"] * 100
print(f"Accuracy: {accuracy:.1f}%")
print(f"Total reward: {reward:.2f}")

env.close()
```

### Custom Training Loop

```python
import gymnasium as gym
import gymnasium_retinatask

env = gym.make(
    "RetinaTask-v0",
    mode="batch",
    batch_size=50,
    reward_type="simple"
)

num_episodes = 100

for episode in range(num_episodes):
    obs, info = env.reset()
    episode_reward = 0

    while True:
        # Your learning algorithm here
        action = your_policy(obs)

        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        # Update policy
        your_policy.update(obs, action, reward)

        if terminated or truncated:
            break

    if episode % 10 == 0:
        print(f"Episode {episode}: Reward = {episode_reward:.2f}")

env.close()
```

## See Also

- {doc}`patterns` - Pattern validation utilities
- {doc}`../content/basic_usage` - Basic usage guide
- {doc}`../tutorials/simple_agent` - Complete examples
