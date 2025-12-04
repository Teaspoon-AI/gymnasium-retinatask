# Pattern Analysis

Understanding the pattern structure is crucial for designing and evaluating algorithms on the Retina Task.

## Pattern Representation

Each pattern is represented as an 8-bit integer (0-255), where each bit corresponds to one retina pixel:

```
Bit:   7  6  5  4  3  2  1  0
      ┌──┬──┬──┬──┬──┬──┬──┬──┐
      │  │  │  │  │  │  │  │  │
      └──┴──┴──┴──┴──┴──┴──┴──┘
       └────Left────┘ └───Right──┘
         Bits 7-4      Bits 3-0
```

### Example Patterns

**Pattern 255 (0b11111111):**
- All pixels ON
- Left: 0b1111 (valid)
- Right: 0b1111 (valid)
- Labels: (1.0, 1.0)

**Pattern 0 (0b00000000):**
- All pixels OFF
- Left: 0b0000 (invalid)
- Right: 0b0000 (invalid)
- Labels: (0.0, 0.0)

## Pattern Distribution

The 256 possible patterns are evenly distributed across four categories:

| Category | Count | Percentage | Left Valid | Right Valid |
|----------|-------|------------|------------|-------------|
| Both valid | 64 | 25% | ✓ | ✓ |
| Only left valid | 64 | 25% | ✓ | ✗ |
| Only right valid | 64 | 25% | ✗ | ✓ |
| Neither valid | 64 | 25% | ✗ | ✗ |

This balanced distribution ensures that the task tests independent classification on both sides.

## Valid Patterns

### Right Side Valid Patterns

The 8 valid patterns for the right side (bits 3-0):

```python
[0b1011, 0b0111, 0b1110, 0b1101, 0b0010, 0b0001, 0b0011, 0b1111]
```

Visualized:
```
Pattern 0b1011:  █ · █ █
Pattern 0b0111:  · █ █ █
Pattern 0b1110:  █ █ █ ·
Pattern 0b1101:  █ █ · █
Pattern 0b0010:  · · █ ·
Pattern 0b0001:  · · · █
Pattern 0b0011:  · · █ █
Pattern 0b1111:  █ █ █ █
```

### Left Side Valid Patterns

The 8 valid patterns for the left side (bits 7-4):

```python
[0b1000, 0b0100, 0b1100, 0b1111, 0b1011, 0b0111, 0b1110, 0b1101]
```

## Using the Pattern API

The `RetinaPatterns` class provides utilities for working with patterns:

```python
from gymnasium_retinatask import RetinaPatterns

# Check if sides are valid
pattern = 255  # 0b11111111
left_valid = RetinaPatterns.is_left_valid(pattern)
right_valid = RetinaPatterns.is_right_valid(pattern)
print(f"Left valid: {left_valid}, Right valid: {right_valid}")
# Output: Left valid: True, Right valid: True

# Get classification labels
left_label, right_label = RetinaPatterns.get_labels(pattern)
print(f"Labels: ({left_label}, {right_label})")
# Output: Labels: (1.0, 1.0)

# Convert to observation
obs = RetinaPatterns.pattern_to_observation(pattern)
print(f"Observation: {obs}")
# Output: Observation: [1. 1. 1. 1. 1. 1. 1. 1.]
```

## Analyzing All Patterns

Run the pattern analysis example to see all pattern types:

```bash
uv run python src/gymnasium_retinatask/examples/pattern_analysis.py
```

Example output:

```
Pattern Distribution:
  Both sides valid:      64 / 256 (25.0%)
  Only left valid:       64 / 256 (25.0%)
  Only right valid:      64 / 256 (25.0%)
  Neither side valid:    64 / 256 (25.0%)
  Total:                256

Example Patterns

Both valid:
Pattern 65 (0b01000001):
  Left  | Right
--------+-------
  ·█  |  ··
  ··  |  ·█
Labels: Left=1, Right=1
```

## Pattern Independence

The key insight is that left and right classifications are **completely independent**:

- A valid left pattern can appear with any right pattern (valid or invalid)
- A valid right pattern can appear with any left pattern (valid or invalid)

This independence is what makes the task a good test for modularity:
- An optimal network should have separate modules for left and right
- The modules should not share information or interfere
- Each module can be optimized independently

## Modularity Metrics

To measure modularity, you can analyze your evolved networks:

1. **Connection Analysis**: Check if connections cross between left and right processing
2. **Ablation Study**: Remove left connections and test right classification
3. **Transfer Learning**: Train on left-only patterns, test on both

Example ablation analysis:

```python
def test_modularity(agent, env):
    """Test if left and right are independently processed."""

    # Test 1: Random left, correct right
    left_errors = []
    right_errors = []

    env = gym.make("RetinaTask-v0", mode="full_evaluation")
    obs, info = env.reset()

    while True:
        pattern = info["pattern"]
        correct_left, correct_right = RetinaPatterns.get_labels(pattern)

        # Agent makes prediction
        action = agent(obs)

        # Record errors
        left_error = abs(action[0] - correct_left)
        right_error = abs(action[1] - correct_right)

        left_errors.append(left_error)
        right_errors.append(right_error)

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated:
            break

    # If modular, errors should be uncorrelated
    correlation = np.corrcoef(left_errors, right_errors)[0, 1]
    print(f"Error correlation: {correlation:.3f}")
    # Low correlation (close to 0) suggests modularity
```

## Expected Performance

### Random Agent

A random agent outputs uniform random values in [0, 1]:

- Expected error per output: 0.5
- Expected error per pattern: 1.0 (0.5 + 0.5)
- Expected total error (256 patterns): 256
- Expected reward (paper fitness): 1000 / (1 + 256) ≈ 3.9

### Perfect Baseline

A perfect classifier:

- Error per pattern: 0.0
- Total error (256 patterns): 0.0
- Reward (paper fitness): 1000 / (1 + 0) = 1000.0

### Learning Progress

Track these metrics during training:

1. **Total reward**: Should increase toward 1000
2. **Classification accuracy**: Percentage of correct classifications
3. **Modularity**: Correlation between left and right errors (should be low)
4. **Generalization**: Performance on held-out patterns

## Visualization

Visualize pattern structure in 2D:

```python
import matplotlib.pyplot as plt
import numpy as np
from gymnasium_retinatask import RetinaPatterns

# Create a 16x16 grid (256 patterns)
grid = np.zeros((16, 16))

for pattern in range(256):
    left_valid = RetinaPatterns.is_left_valid(pattern)
    right_valid = RetinaPatterns.is_right_valid(pattern)

    # Color code: 0=neither, 1=left, 2=right, 3=both
    value = int(left_valid) + 2 * int(right_valid)

    row = pattern // 16
    col = pattern % 16
    grid[row, col] = value

plt.imshow(grid, cmap='viridis')
plt.colorbar(label='0=neither, 1=left, 2=right, 3=both')
plt.title('Pattern Classification Grid')
plt.xlabel('Pattern (mod 16)')
plt.ylabel('Pattern // 16')
plt.show()
```

This visualization shows the structured nature of the pattern space and can help understand how your algorithm explores it.
