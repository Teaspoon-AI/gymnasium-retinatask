---
hide-toc: true
firstpage:
lastpage:
---

```{project-logo} _static/img/retinatask-text.png
:alt: Retina Task Logo
```

```{project-heading}
A Gymnasium-compatible implementation of the **Left & Right Retina Problem**, a benchmark task for testing modular neural network evolution.
```

The Retina Task is a pattern classification problem originally introduced by Kashtan and Alon (2005) and used extensively in research on evolving modular neural networks, including the ES-HyperNEAT algorithm (Risi & Stanley, Artificial Life 2012).

This implementation provides:
- ✅ **Clean, ML-agnostic API** - Compatible with any learning algorithm
- ✅ **Gymnasium standard** - Follows Farama Foundation conventions
- ✅ **Multiple evaluation modes** - Single pattern, batch, or full evaluation
- ✅ **Well-tested** - Comprehensive test suite included
- ✅ **Easy to use** - Simple installation and straightforward API

## Quick Start

Install with uv or pip:

```bash
# Using uv (recommended)
uv pip install gymnasium-retinatask

# Using pip
pip install gymnasium-retinatask
```

Basic usage:

```python
import gymnasium as gym
import gymnasium_retinatask

# Create environment
env = gym.make("RetinaTask-v0")

# Run one episode
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

env.close()
```

## The Task

The artificial retina consists of 8 pixels in a 4×2 grid:

```
┌─────────────┬─────────────┐
│  Left (4)   │  Right (4)  │
│   pixels    │   pixels    │
└─────────────┴─────────────┘
```

The agent must independently classify whether the pattern on each side is valid:
- **Left output**: 1.0 = valid pattern, 0.0 = invalid
- **Right output**: 1.0 = valid pattern, 0.0 = invalid

### Why This Task?

This task tests **modularity** in neural networks. The left and right classification problems should be solved by separate, independent functional modules. This makes it an excellent benchmark for:

- 🧬 Neuroevolution (NEAT, HyperNEAT, ES-HyperNEAT)
- 🏗️ Modular architecture learning
- 🔍 Structural learning algorithms
- 📊 Quality-diversity methods

## Key Features

### Multiple Evaluation Modes

```python
# Single pattern per episode (default)
env = gym.make("RetinaTask-v0", mode="single_pattern")

# Batch of random patterns
env = gym.make("RetinaTask-v0", mode="batch", batch_size=100)

# All 256 patterns in sequence
env = gym.make("RetinaTask-v0", mode="full_evaluation")
```

### Flexible Reward Functions

```python
# Paper fitness function (default): 1000.0 / (1.0 + error)
env = gym.make("RetinaTask-v0", reward_type="paper")

# Simple negative error
env = gym.make("RetinaTask-v0", reward_type="simple")
```

## Pattern Distribution

Out of 256 possible patterns (2^8):
- 25% have both sides valid
- 25% have only left valid
- 25% have only right valid
- 25% have neither side valid

## Citation

If you use this environment in your research, please cite:

```bibtex
@software{gymnasium-retinatask,
  author = {Stefano Palmieri},
  title = {Gymnasium Retina Task: A Benchmark for Modular Neural Networks},
  url = {https://github.com/Teaspoon-AI/gymnasium-retinatask},
  year = {2025},
}
```

**Original task reference:**

```bibtex
@article{Risi2012AnEH,
  title={An Enhanced Hypercube-Based Encoding for Evolving the Placement, Density, and Connectivity of Neurons},
  author={Sebastian Risi and Kenneth O. Stanley},
  journal={Artificial Life},
  year={2012},
  volume={18},
  pages={331-363},
  url={https://api.semanticscholar.org/CorpusID:3256786}
}
```

```{toctree}
:hidden:
:caption: User Guide

content/installation
content/basic_usage
content/pattern_analysis
tutorials/simple_agent
tutorials/evaluation
```

```{toctree}
:hidden:
:caption: API Reference

api/environment
api/patterns
```

```{toctree}
:hidden:
:caption: Development

release_notes
content/contributing
GitHub <https://github.com/Teaspoon-AI/gymnasium-retinatask>
```
