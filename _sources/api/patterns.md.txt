# Pattern Utilities API

## RetinaPatterns

```{eval-rst}
.. autoclass:: gymnasium_retinatask.retina_env.RetinaPatterns
   :members:
   :undoc-members:
   :show-inheritance:
```

## Class Methods

### is_left_valid

```python
@staticmethod
def is_left_valid(pattern: int) -> bool
```

Check if the left side (bits 7-4) of the pattern is valid.

**Parameters:**
- `pattern` (int): 8-bit integer (0-255) representing the retina state

**Returns:**
- `bool`: True if the left 4 bits match a valid left pattern

**Example:**

```python
from gymnasium_retinatask import RetinaPatterns

# Pattern with left side 0b1111 (valid)
pattern = 0b11110000  # 240
assert RetinaPatterns.is_left_valid(pattern) == True

# Pattern with left side 0b0000 (invalid)
pattern = 0b00001111  # 15
assert RetinaPatterns.is_left_valid(pattern) == False
```

### is_right_valid

```python
@staticmethod
def is_right_valid(pattern: int) -> bool
```

Check if the right side (bits 3-0) of the pattern is valid.

**Parameters:**
- `pattern` (int): 8-bit integer (0-255) representing the retina state

**Returns:**
- `bool`: True if the right 4 bits match a valid right pattern

**Example:**

```python
from gymnasium_retinatask import RetinaPatterns

# Pattern with right side 0b1111 (valid)
pattern = 0b00001111  # 15
assert RetinaPatterns.is_right_valid(pattern) == True

# Pattern with right side 0b0000 (invalid)
pattern = 0b11110000  # 240
assert RetinaPatterns.is_right_valid(pattern) == False
```

### get_labels

```python
@staticmethod
def get_labels(pattern: int) -> Tuple[float, float]
```

Get the correct classification labels for a pattern.

**Parameters:**
- `pattern` (int): 8-bit integer (0-255) representing the retina state

**Returns:**
- `Tuple[float, float]`: `(left_label, right_label)` where each is 1.0 if valid, 0.0 if invalid

**Example:**

```python
from gymnasium_retinatask import RetinaPatterns

# Both sides valid
pattern = 0b11111111  # 255
left, right = RetinaPatterns.get_labels(pattern)
print(f"Left: {left}, Right: {right}")  # Left: 1.0, Right: 1.0

# Only left valid
pattern = 0b11110000  # 240
left, right = RetinaPatterns.get_labels(pattern)
print(f"Left: {left}, Right: {right}")  # Left: 1.0, Right: 0.0

# Neither valid
pattern = 0b00000000  # 0
left, right = RetinaPatterns.get_labels(pattern)
print(f"Left: {left}, Right: {right}")  # Left: 0.0, Right: 0.0
```

### pattern_to_observation

```python
@staticmethod
def pattern_to_observation(pattern: int) -> np.ndarray
```

Convert an 8-bit pattern to an observation array.

**Parameters:**
- `pattern` (int): 8-bit integer (0-255) representing the retina state

**Returns:**
- `np.ndarray`: Array of shape `(8,)` with dtype `float32`, containing values in {0.0, 1.0}

**Bit Order:**
- Index 0 corresponds to bit 7 (leftmost pixel)
- Index 7 corresponds to bit 0 (rightmost pixel)

**Example:**

```python
import numpy as np
from gymnasium_retinatask import RetinaPatterns

# All pixels on
pattern = 0b11111111  # 255
obs = RetinaPatterns.pattern_to_observation(pattern)
print(obs)  # [1. 1. 1. 1. 1. 1. 1. 1.]

# Alternating pattern
pattern = 0b10101010  # 170
obs = RetinaPatterns.pattern_to_observation(pattern)
print(obs)  # [1. 0. 1. 0. 1. 0. 1. 0.]

# Custom pattern
pattern = 0b11001010  # 202
obs = RetinaPatterns.pattern_to_observation(pattern)
print(obs)  # [1. 1. 0. 0. 1. 0. 1. 0.]
```

## Pattern Constants

### Valid Right Patterns

```python
RIGHT_PATTERNS = [0b1011, 0b0111, 0b1110, 0b1101, 0b0010, 0b0001, 0b0011, 0b1111]
```

The 8 valid 4-bit patterns for the right side of the retina.

### Valid Left Patterns

```python
LEFT_PATTERNS = [0b1000, 0b0100, 0b1100, 0b1111, 0b1011, 0b0111, 0b1110, 0b1101]
```

The 8 valid 4-bit patterns for the left side of the retina.

## Complete Example

```python
import numpy as np
from gymnasium_retinatask import RetinaPatterns

def analyze_pattern(pattern: int):
    """Analyze a pattern and print all information."""
    print(f"\nPattern: {pattern} (0b{pattern:08b})")

    # Check validity
    left_valid = RetinaPatterns.is_left_valid(pattern)
    right_valid = RetinaPatterns.is_right_valid(pattern)
    print(f"Left valid: {left_valid}")
    print(f"Right valid: {right_valid}")

    # Get labels
    left_label, right_label = RetinaPatterns.get_labels(pattern)
    print(f"Labels: ({left_label}, {right_label})")

    # Get observation
    obs = RetinaPatterns.pattern_to_observation(pattern)
    print(f"Observation: {obs}")

    # Visualize
    left_pixels = obs[:4]
    right_pixels = obs[4:]
    print(f"Left pixels:  {''.join('█' if p else '·' for p in left_pixels)}")
    print(f"Right pixels: {''.join('█' if p else '·' for p in right_pixels)}")

# Analyze some patterns
for pattern in [0, 65, 170, 255]:
    analyze_pattern(pattern)
```

Output:

```
Pattern: 0 (0b00000000)
Left valid: False
Right valid: False
Labels: (0.0, 0.0)
Observation: [0. 0. 0. 0. 0. 0. 0. 0.]
Left pixels:  ····
Right pixels: ····

Pattern: 65 (0b01000001)
Left valid: True
Right valid: True
Labels: (1.0, 1.0)
Observation: [0. 1. 0. 0. 0. 0. 0. 1.]
Left pixels:  ·█··
Right pixels: ···█

...
```

## See Also

- {doc}`environment` - Main environment API
- {doc}`../content/pattern_analysis` - Pattern analysis guide
