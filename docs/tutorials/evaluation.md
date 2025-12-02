# Evaluation Tutorial

This tutorial shows how to comprehensively evaluate agents on the Retina Task.

## Basic Evaluation

### Single Pattern Evaluation

Quick test on individual patterns:

```python
import gymnasium as gym
import numpy as np
import gymnasium_retinatask

def evaluate_on_single_pattern(agent, seed=42):
    """Evaluate agent on a single random pattern."""
    env = gym.make("RetinaTask-v0", mode="single_pattern")
    obs, info = env.reset(seed=seed)

    pattern = info["pattern"]
    action = agent(obs)

    obs, reward, terminated, truncated, info = env.step(action)

    print(f"Pattern: {pattern} (0b{pattern:08b})")
    print(f"Action: [{action[0]:.3f}, {action[1]:.3f}]")
    print(f"Correct: [{info['correct_left']:.1f}, {info['correct_right']:.1f}]")
    print(f"Error: {info['pattern_error']:.3f}")
    print(f"Reward: {reward:.2f}")

    env.close()
    return reward

# Example with random agent
evaluate_on_single_pattern(lambda obs: np.random.rand(2))
```

### Batch Evaluation

Evaluate on multiple random patterns:

```python
import gymnasium as gym
import numpy as np
import gymnasium_retinatask

def evaluate_on_batch(agent, batch_size=100, num_trials=10):
    """Evaluate agent on multiple batches."""
    env = gym.make("RetinaTask-v0", mode="batch", batch_size=batch_size)

    rewards = []
    accuracies = []

    for trial in range(num_trials):
        obs, info = env.reset()
        total_reward = 0
        correct = 0

        while True:
            action = agent(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if info["pattern_error"] < 0.5:
                correct += 1

            if terminated or truncated:
                break

        accuracy = correct / batch_size * 100
        rewards.append(total_reward)
        accuracies.append(accuracy)

    env.close()

    print(f"Batch size: {batch_size}")
    print(f"Num trials: {num_trials}")
    print(f"Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Mean accuracy: {np.mean(accuracies):.1f}% ± {np.std(accuracies):.1f}%")

    return rewards, accuracies

# Example
rewards, accs = evaluate_on_batch(
    lambda obs: np.random.rand(2),
    batch_size=100,
    num_trials=10
)
```

### Full Evaluation

Comprehensive test on all 256 patterns:

```python
import gymnasium as gym
import numpy as np
import gymnasium_retinatask
from gymnasium_retinatask import RetinaPatterns

def full_evaluation(agent):
    """Evaluate agent on all 256 patterns."""
    env = gym.make("RetinaTask-v0", mode="full_evaluation")
    obs, info = env.reset()

    # Track statistics
    pattern_errors = []
    left_errors = []
    right_errors = []
    by_category = {
        "both": {"total": 0, "correct": 0},
        "left": {"total": 0, "correct": 0},
        "right": {"total": 0, "correct": 0},
        "neither": {"total": 0, "correct": 0},
    }

    while True:
        pattern = info["pattern"]
        action = agent(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        # Record errors
        pattern_errors.append(info["pattern_error"])
        left_error = abs(action[0] - info["correct_left"])
        right_error = abs(action[1] - info["correct_right"])
        left_errors.append(left_error)
        right_errors.append(right_error)

        # Categorize
        left_valid = info["correct_left"] == 1.0
        right_valid = info["correct_right"] == 1.0

        if left_valid and right_valid:
            category = "both"
        elif left_valid:
            category = "left"
        elif right_valid:
            category = "right"
        else:
            category = "neither"

        by_category[category]["total"] += 1
        if info["pattern_error"] < 0.5:
            by_category[category]["correct"] += 1

        if terminated or truncated:
            total_reward = reward
            break

    env.close()

    # Print results
    print("=" * 60)
    print("Full Evaluation Results")
    print("=" * 60)

    overall_accuracy = (256 - sum(e > 0.5 for e in pattern_errors)) / 256 * 100
    print(f"Overall accuracy: {overall_accuracy:.1f}%")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Mean error: {np.mean(pattern_errors):.3f}")

    print("\nBy category:")
    for cat, stats in by_category.items():
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {cat:8s}: {stats['correct']:3d}/{stats['total']:3d} = {acc:5.1f}%")

    print(f"\nLeft side accuracy:  {(64 - sum(e > 0.5 for e in left_errors)) / 256 * 100:.1f}%")
    print(f"Right side accuracy: {(64 - sum(e > 0.5 for e in right_errors)) / 256 * 100:.1f}%")

    # Check modularity (error correlation)
    correlation = np.corrcoef(left_errors, right_errors)[0, 1]
    print(f"\nError correlation: {correlation:.3f}")
    print("  (Low correlation suggests modular processing)")

    return {
        "accuracy": overall_accuracy,
        "reward": total_reward,
        "mean_error": np.mean(pattern_errors),
        "by_category": by_category,
        "correlation": correlation,
    }

# Example
results = full_evaluation(lambda obs: np.random.rand(2))
```

## Advanced Evaluation

### Cross-Validation

Evaluate with different random seeds:

```python
import gymnasium as gym
import numpy as np

def cross_validation(agent, num_folds=5):
    """Evaluate agent with multiple seeds."""
    results = []

    for fold in range(num_folds):
        env = gym.make("RetinaTask-v0", mode="full_evaluation")
        obs, info = env.reset(seed=fold)

        correct = 0
        while True:
            action = agent(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            if info["pattern_error"] < 0.5:
                correct += 1

            if terminated or truncated:
                break

        env.close()

        accuracy = correct / 256 * 100
        results.append(accuracy)
        print(f"Fold {fold + 1}: {accuracy:.1f}%")

    print(f"\nCross-validation accuracy: {np.mean(results):.1f}% ± {np.std(results):.1f}%")
    return results
```

### Performance Profile

Analyze performance across pattern types:

```python
import gymnasium as gym
import numpy as np
from gymnasium_retinatask import RetinaPatterns

def performance_profile(agent):
    """Detailed performance analysis."""
    env = gym.make("RetinaTask-v0", mode="full_evaluation")
    obs, info = env.reset()

    # Track by pattern properties
    by_left_bits = {i: {"total": 0, "correct": 0} for i in range(5)}
    by_right_bits = {i: {"total": 0, "correct": 0} for i in range(5)}

    while True:
        pattern = info["pattern"]
        action = agent(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        # Count bits
        left_bits = bin(pattern >> 4).count('1')
        right_bits = bin(pattern & 0b1111).count('1')

        left_correct = abs(action[0] - info["correct_left"]) < 0.5
        right_correct = abs(action[1] - info["correct_right"]) < 0.5

        by_left_bits[left_bits]["total"] += 1
        if left_correct:
            by_left_bits[left_bits]["correct"] += 1

        by_right_bits[right_bits]["total"] += 1
        if right_correct:
            by_right_bits[right_bits]["correct"] += 1

        if terminated or truncated:
            break

    env.close()

    # Print profile
    print("Left side accuracy by bit count:")
    for bits in range(5):
        stats = by_left_bits[bits]
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {bits} bits: {stats['correct']:2d}/{stats['total']:2d} = {acc:5.1f}%")

    print("\nRight side accuracy by bit count:")
    for bits in range(5):
        stats = by_right_bits[bits]
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {bits} bits: {stats['correct']:2d}/{stats['total']:2d} = {acc:5.1f}%")
```

### Learning Curve

Track performance during training:

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(agent_fn, num_epochs=100):
    """Plot learning curve over training."""
    env = gym.make("RetinaTask-v0", mode="full_evaluation")

    accuracies = []
    rewards = []

    for epoch in range(num_epochs):
        obs, info = env.reset()
        correct = 0
        epoch_reward = 0

        while True:
            action = agent_fn(obs, training=True)  # Training mode
            obs, reward, terminated, truncated, info = env.step(action)
            epoch_reward += reward

            if info["pattern_error"] < 0.5:
                correct += 1

            if terminated or truncated:
                break

        accuracy = correct / 256 * 100
        accuracies.append(accuracy)
        rewards.append(epoch_reward)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: Accuracy = {accuracy:.1f}%")

    env.close()

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(accuracies)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Learning Curve - Accuracy')
    ax1.grid(True)

    ax2.plot(rewards)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Reward')
    ax2.set_title('Learning Curve - Reward')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('learning_curve.png')
    plt.show()

    return accuracies, rewards
```

## Benchmarking

Compare multiple agents:

```python
import gymnasium as gym
import numpy as np

def benchmark_agents(agents_dict):
    """Benchmark multiple agents."""
    results = {}

    for name, agent in agents_dict.items():
        print(f"\nEvaluating {name}...")
        env = gym.make("RetinaTask-v0", mode="full_evaluation")
        obs, info = env.reset()

        correct = 0
        total_reward = 0

        while True:
            action = agent(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if info["pattern_error"] < 0.5:
                correct += 1

            if terminated or truncated:
                break

        env.close()

        accuracy = correct / 256 * 100
        results[name] = {
            "accuracy": accuracy,
            "reward": total_reward,
        }

    # Print comparison
    print("\n" + "=" * 60)
    print("Benchmark Results")
    print("=" * 60)
    print(f"{'Agent':<20} {'Accuracy':<12} {'Reward':<12}")
    print("-" * 60)
    for name, res in results.items():
        print(f"{name:<20} {res['accuracy']:>6.1f}%     {res['reward']:>8.2f}")

    return results

# Example
from gymnasium_retinatask import RetinaPatterns

agents = {
    "Random": lambda obs: np.random.rand(2),
    "Always 0.5": lambda obs: np.array([0.5, 0.5]),
    "Threshold": lambda obs: np.array([
        1.0 if obs[:4].sum() >= 2 else 0.0,
        1.0 if obs[4:].sum() >= 2 else 0.0
    ]),
    "Perfect": lambda obs: np.array(RetinaPatterns.get_labels(
        int(''.join(str(int(b)) for b in obs), 2)
    )),
}

results = benchmark_agents(agents)
```

## See Also

- {doc}`simple_agent` - Simple agent examples
- {doc}`../api/environment` - Environment API reference
- {doc}`../content/pattern_analysis` - Pattern analysis guide
