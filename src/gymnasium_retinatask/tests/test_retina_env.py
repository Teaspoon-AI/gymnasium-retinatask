"""Tests for the Retina Task environment."""

import gymnasium as gym
import numpy as np
import pytest


class TestRetinaPatterns:
    """Test pattern validation logic."""

    def test_imports(self):
        """Test that the module imports correctly."""
        import gymnasium_retinatask  # noqa: F401
        from gymnasium_retinatask import RetinaPatterns

        assert RetinaPatterns is not None

    def test_pattern_validation(self):
        """Test pattern validation for known cases."""
        from gymnasium_retinatask import RetinaPatterns

        # Test a pattern that should have both sides valid
        # Pattern 0b11111111 = 255
        # Left: 0b1111, Right: 0b1111 - both should be valid
        assert RetinaPatterns.is_left_valid(0b11111111)
        assert RetinaPatterns.is_right_valid(0b11111111)

        # Test specific valid patterns from the paper
        # Right pattern 0b1011 should be valid
        assert RetinaPatterns.is_right_valid(0b1011)

        # Left pattern 0b1000 << 4 should be valid
        assert RetinaPatterns.is_left_valid(0b10000000)

    def test_get_labels(self):
        """Test label generation."""
        from gymnasium_retinatask import RetinaPatterns

        # All bits on - both sides valid
        left, right = RetinaPatterns.get_labels(0b11111111)
        assert left == 1.0
        assert right == 1.0

        # Pattern with invalid sides should return 0.0
        left, right = RetinaPatterns.get_labels(0b00000000)
        assert left == 0.0
        assert right == 0.0

    def test_pattern_to_observation(self):
        """Test pattern conversion to observation."""
        from gymnasium_retinatask import RetinaPatterns

        # All zeros
        obs = RetinaPatterns.pattern_to_observation(0b00000000)
        assert np.all(obs == 0.0)
        assert obs.shape == (8,)

        # All ones
        obs = RetinaPatterns.pattern_to_observation(0b11111111)
        assert np.all(obs == 1.0)

        # Specific pattern
        obs = RetinaPatterns.pattern_to_observation(0b10101010)
        expected = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float32)
        assert np.array_equal(obs, expected)


class TestRetinaEnv:
    """Test the Retina environment."""

    def test_env_creation(self):
        """Test environment can be created."""
        import gymnasium_retinatask  # noqa: F401

        env = gym.make("RetinaTask-v0")
        assert env is not None
        env.close()

    def test_observation_space(self):
        """Test observation space is correct."""
        env = gym.make("RetinaTask-v0")
        assert env.observation_space.shape == (8,)
        assert env.observation_space.dtype == np.float32
        env.close()

    def test_action_space(self):
        """Test action space is correct."""
        env = gym.make("RetinaTask-v0")
        assert env.action_space.shape == (2,)
        assert env.action_space.dtype == np.float32
        env.close()

    def test_reset(self):
        """Test environment reset."""
        env = gym.make("RetinaTask-v0")
        obs, info = env.reset(seed=42)

        assert obs.shape == (8,)
        assert "pattern" in info
        assert 0 <= info["pattern"] < 256
        env.close()

    def test_step(self):
        """Test environment step."""
        env = gym.make("RetinaTask-v0")
        obs, info = env.reset(seed=42)

        # Take a random action
        action = np.array([0.5, 0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "pattern_error" in info
        env.close()

    def test_single_pattern_mode(self):
        """Test single pattern mode completes in one step."""
        env = gym.make("RetinaTask-v0", mode="single_pattern")
        obs, info = env.reset(seed=42)

        action = np.array([1.0, 1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        assert terminated
        assert info["patterns_evaluated"] == 1
        env.close()

    def test_batch_mode(self):
        """Test batch mode with multiple patterns."""
        batch_size = 10
        env = gym.make("RetinaTask-v0", mode="batch", batch_size=batch_size)
        obs, info = env.reset(seed=42)

        assert info["total_patterns"] == batch_size

        # Step through all patterns
        for i in range(batch_size):
            action = np.array([0.5, 0.5], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)

            if i < batch_size - 1:
                assert not terminated
            else:
                assert terminated
                assert info["patterns_evaluated"] == batch_size

        env.close()

    def test_full_evaluation_mode(self):
        """Test full evaluation mode with all 256 patterns."""
        env = gym.make("RetinaTask-v0", mode="full_evaluation")
        obs, info = env.reset(seed=42)

        assert info["total_patterns"] == 256

        # Step through all patterns
        for i in range(256):
            action = np.array([0.5, 0.5], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)

            if i < 255:
                assert not terminated
            else:
                assert terminated
                assert info["patterns_evaluated"] == 256

        env.close()

    def test_perfect_classification(self):
        """Test that perfect classification gets high reward."""
        env = gym.make("RetinaTask-v0", mode="single_pattern")

        # Reset with a known seed
        obs, info = env.reset(seed=42)
        pattern = info["pattern"]

        # Get correct labels
        from gymnasium_retinatask import RetinaPatterns

        left_label, right_label = RetinaPatterns.get_labels(pattern)

        # Give perfect classification
        action = np.array([left_label, right_label], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        # With perfect classification, error should be 0
        assert info["pattern_error"] == pytest.approx(0.0)
        # Reward should be maximum (1000.0 / 1.0 = 1000.0)
        assert reward == pytest.approx(1000.0)

        env.close()

    def test_reward_types(self):
        """Test different reward types."""
        # Test paper reward
        env1 = gym.make("RetinaTask-v0", reward_type="paper")
        obs, _ = env1.reset(seed=42)
        action = np.array([0.5, 0.5], dtype=np.float32)
        _, reward1, _, _, _ = env1.step(action)
        assert float(reward1) > 0  # Paper reward is always positive
        env1.close()

        # Test simple reward
        env2 = gym.make("RetinaTask-v0", reward_type="simple")
        obs, _ = env2.reset(seed=42)
        action = np.array([0.5, 0.5], dtype=np.float32)
        _, reward2, _, _, _ = env2.step(action)
        assert float(reward2) <= 0  # Simple reward is negative error
        env2.close()
