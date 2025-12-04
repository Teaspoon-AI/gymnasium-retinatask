# Simple Agent Tutorial

This tutorial shows how to create and evaluate simple agents on the Retina Task.

## Random Agent

The simplest possible agent outputs random classifications:

```python
import gymnasium as gym
import numpy as np
import gymnasium_retinatask

def random_agent_example():
    """Run a random agent for multiple episodes."""
    env = gym.make("RetinaTask-v0", mode="batch", batch_size=100)

    num_episodes = 10
    rewards = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0

        while True:
            # Random action
            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                break

        rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

    env.close()

    print(f"\nAverage reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Expected: ~10 (for random agent)")

if __name__ == "__main__":
    random_agent_example()
```

**Expected performance:**
- Average reward: ~10
- This is the baseline - any learning algorithm should beat this!

## Threshold Agent

A simple thresholding strategy:

```python
import gymnasium as gym
import numpy as np
import gymnasium_retinatask

def threshold_agent_example():
    """Agent that uses a simple threshold on pixel counts."""
    env = gym.make("RetinaTask-v0", mode="full_evaluation")

    obs, info = env.reset()
    total_reward = 0
    correct = 0

    while True:
        # Count pixels on each side
        left_pixels = obs[:4]
        right_pixels = obs[4:]

        left_count = np.sum(left_pixels)
        right_count = np.sum(right_pixels)

        # Simple threshold: classify as valid if >= 2 pixels on
        action = np.array([
            1.0 if left_count >= 2 else 0.0,
            1.0 if right_count >= 2 else 0.0
        ], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if info["pattern_error"] == 0.0:
            correct += 1

        if terminated or truncated:
            break

    env.close()

    accuracy = correct / 256 * 100
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Total reward: {total_reward:.2f}")

if __name__ == "__main__":
    threshold_agent_example()
```

**Expected performance:**
- Accuracy: ~60-70%
- Better than random, but far from optimal

## Lookup Table Agent

The perfect baseline using a lookup table:

```python
import gymnasium as gym
import numpy as np
from gymnasium_retinatask import RetinaPatterns

def lookup_table_agent():
    """Perfect agent using ground truth lookup."""
    env = gym.make("RetinaTask-v0", mode="full_evaluation")

    obs, info = env.reset()
    total_reward = 0

    while True:
        # Use pattern validation (this is "cheating" for a real agent)
        pattern = info["pattern"]
        left_label, right_label = RetinaPatterns.get_labels(pattern)

        action = np.array([left_label, right_label], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    env.close()

    print(f"Accuracy: 100.0%")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Expected: 1000.0")

if __name__ == "__main__":
    lookup_table_agent()
```

**Expected performance:**
- Accuracy: 100%
- Reward: 1000.0
- This is the upper bound for any algorithm

## Simple Neural Network

A basic feedforward neural network (requires PyTorch):

```python
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium_retinatask
from gymnasium_retinatask import RetinaPatterns

class SimpleClassifier(nn.Module):
    """Simple feedforward network for retina classification."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(self, x):
        return self.network(x)


def train_simple_network(num_epochs=100):
    """Train a simple neural network on the retina task."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = SimpleClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    # Create environment
    env = gym.make("RetinaTask-v0", mode="full_evaluation")

    for epoch in range(num_epochs):
        obs, info = env.reset()
        epoch_loss = 0.0
        correct = 0

        while True:
            # Get ground truth labels
            pattern = info["pattern"]
            left_label, right_label = RetinaPatterns.get_labels(pattern)
            labels = torch.tensor([[left_label, right_label]],
                                  dtype=torch.float32, device=device)

            # Forward pass
            obs_tensor = torch.tensor([obs], dtype=torch.float32, device=device)
            predictions = model(obs_tensor)

            # Compute loss
            loss = criterion(predictions, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Take action in environment
            action = predictions[0].detach().cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action)

            if info["pattern_error"] < 0.5:  # Correct if close enough
                correct += 1

            if terminated or truncated:
                break

        if (epoch + 1) % 10 == 0:
            accuracy = correct / 256 * 100
            print(f"Epoch {epoch + 1}: Loss = {epoch_loss:.4f}, "
                  f"Accuracy = {accuracy:.1f}%")

    env.close()

    # Final evaluation
    env = gym.make("RetinaTask-v0", mode="full_evaluation")
    obs, info = env.reset()
    correct = 0
    total_reward = 0

    model.eval()
    with torch.no_grad():
        while True:
            obs_tensor = torch.tensor([obs], dtype=torch.float32, device=device)
            predictions = model(obs_tensor)
            action = predictions[0].cpu().numpy()

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if info["pattern_error"] < 0.5:
                correct += 1

            if terminated or truncated:
                break

    env.close()

    final_accuracy = correct / 256 * 100
    print(f"\nFinal Accuracy: {final_accuracy:.1f}%")
    print(f"Final Reward: {total_reward:.2f}")

    return model


if __name__ == "__main__":
    model = train_simple_network(num_epochs=100)
```

**Expected performance (after training):**
- Accuracy: 90-100%
- Reward: 500-1000
- Training time: < 1 minute

## Key Takeaways

1. **Random baseline** (~10 reward) - easy to beat
2. **Simple heuristics** can get 60-70% accuracy
3. **Perfect baseline** (1000 reward) - upper bound
4. **Neural networks** can learn to solve this task well

## Next Steps

- Try different network architectures
- Implement evolutionary algorithms
- Analyze learned modularity
- See {doc}`evaluation` for comprehensive evaluation strategies

## See Also

- {doc}`evaluation` - Comprehensive evaluation tutorial
- {doc}`../api/environment` - Environment API reference
- {doc}`../content/pattern_analysis` - Understanding patterns
