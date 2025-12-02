# Claude Code Development Guide

This document describes how this project was developed with Claude Code and provides guidance for future development.

## Project Overview

**gymnasium-retinatask** is a Gymnasium-compatible implementation of the Left & Right Retina Problem, created entirely using Claude Code (Sonnet 4.5) on December 1, 2025.

## Development Process

### Initial Requirements

The project was created from:
- A PDF describing the Retina Task (risi_gecco11.pdf)
- A reference implementation (gymnasium-hardmaze)
- A starter code file with evolutionary algorithm coupling (retinatask.py)

**Key Requirements:**
- Follow Farama Foundation standards
- Use `uv` as package manager
- Create a clean, ML-agnostic implementation
- Decouple from evolutionary algorithms
- Comprehensive documentation

### Development Timeline

1. **Environment Design** (30 minutes)
   - Analyzed the PDF to understand the task
   - Reviewed gymnasium-hardmaze for structural reference
   - Designed clean API separating task logic from algorithms

2. **Core Implementation** (45 minutes)
   - Implemented `RetinaPatterns` for pattern validation
   - Created `RetinaEnvV0` following Gymnasium API
   - Added multiple evaluation modes (single, batch, full)
   - Implemented both paper and simple reward functions

3. **Testing & Examples** (30 minutes)
   - Created comprehensive test suite (14 tests)
   - Developed example scripts (random, perfect, analysis)
   - Verified all tests pass

4. **Documentation** (90 minutes)
   - Set up Sphinx with Furo theme
   - Created user guide, API reference, and tutorials
   - Built and verified HTML documentation
   - Fixed encoding issues in README

5. **Refinements** (15 minutes)
   - Updated citations to correct paper
   - Fixed special character encoding
   - Final verification

## Project Structure

```
gymnasium-retinatask/
├── src/gymnasium_retinatask/
│   ├── __init__.py              # Package initialization & registration
│   ├── retina_env.py            # Main environment implementation
│   ├── examples/                # Example scripts
│   │   ├── random_agent.py
│   │   ├── perfect_agent.py
│   │   └── pattern_analysis.py
│   └── tests/                   # Test suite
│       └── test_retina_env.py
├── docs/                        # Sphinx documentation
│   ├── conf.py
│   ├── index.md
│   ├── api/                     # API reference
│   ├── content/                 # User guides
│   └── tutorials/               # Tutorials
├── pyproject.toml               # Project configuration (uv/hatchling)
├── README.md
├── LICENSE
└── CLAUDE.md                    # This file
```

## Key Design Decisions

### 1. ML-Agnostic Architecture

**Problem:** Original code was tightly coupled to evolutionary algorithms.

**Solution:**
- Separated pattern validation logic into `RetinaPatterns` class
- Made environment independent of learning algorithm
- Provided multiple evaluation modes for different use cases

### 2. Multiple Evaluation Modes

**Rationale:** Different algorithms have different evaluation needs.

**Modes:**
- `single_pattern`: Quick testing, online learning
- `batch`: Stochastic batch evaluation
- `full_evaluation`: Comprehensive benchmarking (all 256 patterns)

### 3. Flexible Reward Functions

**Options:**
- `paper`: Original fitness function `1000.0 / (1.0 + error)`
- `simple`: Negative error for gradient-based methods

### 4. Pattern Representation

**Choice:** 8-bit integers (0-255)

**Benefits:**
- Compact representation
- Easy bit manipulation for validation
- Clear mapping to observation space
- All 256 patterns enumerable

## Working with Claude Code

### Effective Prompting Strategies

1. **Start with Clear Context**
   ```
   "Create a gymnasium environment for the Retina Task as described in
   pdf file. Follow Farama Foundation standards and use uv."
   ```

2. **Reference Existing Patterns**
   ```
   "Use ~/Desktop/gymnasium-hardmaze as a reference for structure"
   ```

3. **Specify Constraints**
   ```
   "The implementation should be clean and agnostic to the ML algorithm"
   ```

### What Claude Did Well

✅ **Understood Domain Requirements**
- Correctly interpreted the PDF description
- Identified key task properties (modularity, independence)
- Maintained fidelity to original paper

✅ **Followed Standards**
- Proper Gymnasium API implementation
- Farama Foundation conventions
- Clean project structure

✅ **Comprehensive Testing**
- Unit tests for all components
- Example scripts for validation
- Documentation with examples

✅ **Documentation**
- Clear API reference
- Practical tutorials
- Professional Sphinx docs with Furo theme

### Areas for Human Review

⚠️ **Domain Expertise**
- Verify pattern validation logic matches paper
- Confirm reward function implementation
- Check evaluation mode semantics

⚠️ **Design Choices**
- Review API design decisions
- Validate performance characteristics
- Consider additional features

### Modifying Existing Code

**Example: Change reward function**

```python
# Prompt:
"Add a new reward type 'sparse' that only gives reward when both
left and right classifications are correct. Update tests and docs."
```

**Claude will:**
1. Modify `RetinaEnvV0.step()`
2. Add new reward calculation
3. Update tests
4. Update API documentation

### Debugging

**Example: Performance issue**

```python
# Prompt:
"The full_evaluation mode is slow with large batches. Profile the code
and optimize the bottlenecks."
```

**Claude will:**
1. Analyze code for bottlenecks
2. Suggest optimizations
3. Implement changes
4. Verify performance improvement

## Common Development Tasks

### Running Tests

```bash
# All tests
uv run pytest

# Specific test
uv run pytest src/gymnasium_retinatask/tests/test_retina_env.py::TestRetinaEnv::test_reset

# With coverage
uv run pytest --cov=gymnasium_retinatask
```

### Building Documentation

```bash
# Build HTML docs
cd docs
uv run sphinx-build -b html . _build/html

# Auto-rebuild on changes
uv run sphinx-autobuild . _build/html

# Open in browser
open _build/html/index.html
```

### Code Formatting

```bash
# Format code
uv run black src/

# Sort imports
uv run isort src/

# Type check
uv run pyright src/
```

### Adding Dependencies

```bash
# Add runtime dependency
uv add numpy

# Add dev dependency
uv add --dev pytest-cov

# Sync environment
uv sync --all-groups
```

## Integration Examples

### With PyTorch

```python
import gymnasium as gym
import torch
import torch.nn as nn
from gymnasium_retinatask import RetinaPatterns

class RetinaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Training loop
env = gym.make("RetinaTask-v0", mode="full_evaluation")
model = RetinaClassifier()
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.BCELoss()

obs, info = env.reset()
while True:
    # Get ground truth
    pattern = info["pattern"]
    left, right = RetinaPatterns.get_labels(pattern)
    target = torch.tensor([left, right])

    # Forward pass
    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    pred = model(obs_tensor)

    # Backward pass
    loss = criterion(pred, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Step environment
    action = pred.detach().numpy()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated:
        break
```

### With NEAT/HyperNEAT

```python
import gymnasium as gym
import neat  # neat-python

def eval_genome(genome, config):
    """Evaluate a NEAT genome on the Retina Task."""
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = gym.make("RetinaTask-v0", mode="full_evaluation")

    obs, info = env.reset()
    total_reward = 0

    while True:
        # NEAT network expects list input
        action = net.activate(obs.tolist())
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated:
            break

    return total_reward

# Run NEAT
config = neat.Config(...)
pop = neat.Population(config)
winner = pop.run(eval_genome, 100)
```

### With JAX

```python
import gymnasium as gym
import jax
import jax.numpy as jnp
from gymnasium_retinatask import RetinaPatterns

@jax.jit
def forward(params, x):
    """Simple JAX network."""
    w1, b1, w2, b2 = params
    h = jax.nn.relu(x @ w1 + b1)
    return jax.nn.sigmoid(h @ w2 + b2)

# Training
env = gym.make("RetinaTask-v0", mode="batch", batch_size=100)
# ... JAX training loop
```

## Tips for Extending

### Adding a New Environment Variant

1. **Subclass RetinaEnvV0**
   ```python
   class HarderRetinaEnv(RetinaEnvV0):
       """Retina task with larger grid."""
       def __init__(self, grid_size=6, **kwargs):
           self.grid_size = grid_size
           super().__init__(**kwargs)
   ```

2. **Register the variant**
   ```python
   register(
       id="HarderRetinaTask-v0",
       entry_point="gymnasium_retinatask:HarderRetinaEnv",
   )
   ```

3. **Add tests**
   ```python
   def test_harder_variant():
       env = gym.make("HarderRetinaTask-v0")
       assert env.observation_space.shape == (12,)  # 6x2 grid
   ```

### Adding Visualization

```python
# Prompt to Claude:
"Add a render() method that visualizes the current retina pattern
using pygame. Show the 8 pixels as a 2x4 grid with valid patterns
highlighted in green."
```

### Performance Optimization

```python
# Prompt to Claude:
"Vectorize the full_evaluation mode to process all 256 patterns
in parallel using numpy broadcasting instead of a loop."
```

## Maintenance

### Updating Dependencies

```bash
# Update all dependencies
uv lock --upgrade

# Update specific package
uv add "gymnasium>=1.1.0"
```

### Testing Across Python Versions

```bash
# Test with different Python versions
uv run --python 3.10 pytest
uv run --python 3.11 pytest
uv run --python 3.12 pytest
```

### Documentation Updates

After making changes:

```bash
# Rebuild docs
cd docs
uv run sphinx-build -b html . _build/html

# Check for warnings
uv run sphinx-build -W -b html . _build/html
```

## Credits

**Created by:** Claude Code (Sonnet 4.5)
**Date:** December 1, 2025
**Developer:** Stefano Palmieri
**Repository:** https://github.com/Teaspoon-AI/gymnasium-retinatask

## References

- **Gymnasium Documentation:** https://gymnasium.farama.org/
- **Farama Foundation:** https://farama.org/
- **Original Paper:** Risi, S., & Stanley, K. O. (2012). An Enhanced Hypercube-Based Encoding for Evolving the Placement, Density, and Connectivity of Neurons. *Artificial Life*, 18(4), 331-363.

---

*This project demonstrates Claude Code's capability to create production-ready Python packages from research papers, following best practices and modern development standards.*
