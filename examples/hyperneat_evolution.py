"""HyperNEAT evolution example for the Retina Task.

This script demonstrates how to evolve neural networks using HyperNEAT
(Hypercube-based NeuroEvolution of Augmenting Topologies) on the Retina Task.

HyperNEAT is particularly well-suited for this task as it can exploit the
geometric structure of the 2D retina to evolve modular solutions.

Requires: neat-python, numpy
Install with: uv pip install neat-python numpy

Note: This is a simplified HyperNEAT implementation for demonstration.
For production use, consider using pureples or other full HyperNEAT libraries.
"""

import gymnasium as gym
import neat
import numpy as np

import gymnasium_retinatask  # noqa: F401


class HyperNEATNetwork:
    """Substrate network created by a HyperNEAT CPPN."""

    def __init__(self, cppn, substrate_config):
        """Initialize HyperNEAT network.

        Args:
            cppn: NEAT network (CPPN) that generates connection weights.
            substrate_config: Configuration for the substrate network.
        """
        self.cppn = cppn
        self.substrate_config = substrate_config

        # Define substrate geometry
        # Input layer: 8 neurons arranged as 2x4 grid (matching retina)
        # Hidden layer: 4 neurons (allowing modularity)
        # Output layer: 2 neurons (left and right classifications)

        self.input_coords = self._create_input_coords()
        self.hidden_coords = self._create_hidden_coords()
        self.output_coords = self._create_output_coords()

        # Generate connection weights using CPPN
        self.weights = self._query_cppn()

    def _create_input_coords(self):
        """Create coordinates for input layer (2x4 grid)."""
        coords = []
        for i in range(2):  # 2 rows
            for j in range(4):  # 4 columns
                x = (j - 1.5) / 2.0  # Normalize to [-1, 1]
                y = (i - 0.5) / 1.0  # Normalize to [-1, 1]
                coords.append((x, y, -1.0))  # z=-1 for input layer
        return coords

    def _create_hidden_coords(self):
        """Create coordinates for hidden layer."""
        # 4 hidden neurons: 2 for left, 2 for right (encouraging modularity)
        return [
            (-0.5, 0.0, 0.0),  # Left module, neuron 1
            (-0.5, 0.5, 0.0),  # Left module, neuron 2
            (0.5, 0.0, 0.0),  # Right module, neuron 1
            (0.5, 0.5, 0.0),  # Right module, neuron 2
        ]

    def _create_output_coords(self):
        """Create coordinates for output layer."""
        return [
            (-0.5, 0.0, 1.0),  # Left output
            (0.5, 0.0, 1.0),  # Right output
        ]

    def _query_cppn(self):
        """Query CPPN to generate substrate connection weights."""
        weights = {
            "input_hidden": np.zeros((len(self.input_coords), len(self.hidden_coords))),
            "hidden_output": np.zeros(
                (len(self.hidden_coords), len(self.output_coords))
            ),
        }

        # Input to hidden connections
        for i, in_coord in enumerate(self.input_coords):
            for j, hid_coord in enumerate(self.hidden_coords):
                # CPPN input: (x1, y1, z1, x2, y2, z2)
                cppn_input = in_coord + hid_coord
                weight = self.cppn.activate(cppn_input)[0]
                weights["input_hidden"][i, j] = weight

        # Hidden to output connections
        for i, hid_coord in enumerate(self.hidden_coords):
            for j, out_coord in enumerate(self.output_coords):
                cppn_input = hid_coord + out_coord
                weight = self.cppn.activate(cppn_input)[0]
                weights["hidden_output"][i, j] = weight

        return weights

    def activate(self, inputs):
        """Activate the substrate network.

        Args:
            inputs: Input activations (8 values).

        Returns:
            Output activations (2 values).
        """
        inputs = np.array(inputs)

        # Input to hidden
        hidden = np.tanh(self.weights["input_hidden"].T @ inputs)

        # Hidden to output
        output = 1.0 / (1.0 + np.exp(-(self.weights["hidden_output"].T @ hidden)))

        return output.tolist()


def eval_genome(genome, config):
    """Evaluate a HyperNEAT genome on the Retina Task.

    Args:
        genome: NEAT genome (CPPN) to evaluate.
        config: NEAT configuration.

    Returns:
        Fitness score (higher is better).
    """
    # Create CPPN from genome
    cppn = neat.nn.FeedForwardNetwork.create(genome, config)

    # Create substrate network
    substrate_net = HyperNEATNetwork(cppn, config)

    # Create environment
    env = gym.make("RetinaTask-v0", mode="full_evaluation", reward_type="paper")

    obs, info = env.reset()
    total_reward = 0.0

    while True:
        # Activate substrate network
        action = substrate_net.activate(obs.tolist())

        obs, reward, terminated, truncated, info = env.step(np.array(action))
        total_reward += float(reward)

        if terminated or truncated:
            break

    env.close()

    return total_reward


def eval_genomes(genomes, config):
    """Evaluate all genomes in a generation.

    Args:
        genomes: List of (genome_id, genome) tuples.
        config: NEAT configuration.
    """
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run_hyperneat(config_path: str, num_generations: int = 100):
    """Run HyperNEAT evolution.

    Args:
        config_path: Path to NEAT configuration file.
        num_generations: Number of generations to evolve.

    Returns:
        Best genome found.
    """
    # Load configuration
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    # Create population
    pop = neat.Population(config)

    # Add reporters
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    # Run evolution
    print("=" * 70)
    print("HyperNEAT Evolution - Retina Task")
    print("=" * 70)
    print("\nHyperNEAT evolves CPPNs that generate substrate network weights.")
    print("The substrate has a geometric structure matching the 2D retina,")
    print("which should encourage modular solutions.\n")
    print(f"Evolving for {num_generations} generations...\n")

    winner = pop.run(eval_genomes, num_generations)

    # Display results
    print("\n" + "=" * 70)
    print("Best Genome (CPPN)")
    print("=" * 70)
    print(f"\nFitness: {winner.fitness:.2f}")
    print(f"CPPN Nodes: {len(winner.nodes)}")
    print(f"CPPN Connections: {len(winner.connections)}")

    return winner, config


def test_winner(winner, config):
    """Test the winning genome and analyze modularity.

    Args:
        winner: Best genome.
        config: NEAT configuration.
    """
    print("\n" + "=" * 70)
    print("Testing Winner")
    print("=" * 70)

    # Create networks
    cppn = neat.nn.FeedForwardNetwork.create(winner, config)
    substrate_net = HyperNEATNetwork(cppn, config)

    # Test on environment
    env = gym.make("RetinaTask-v0", mode="full_evaluation")
    obs, info = env.reset()

    correct_total = 0
    correct_left = 0
    correct_right = 0

    while True:
        action = substrate_net.activate(obs.tolist())

        obs, reward, terminated, truncated, info = env.step(np.array(action))

        if info["pattern"] is not None:
            if info["pattern_error"] == 0.0:
                correct_total += 1
            if abs(action[0] - info["correct_left"]) < 0.5:
                correct_left += 1
            if abs(action[1] - info["correct_right"]) < 0.5:
                correct_right += 1

        if terminated or truncated:
            break

    env.close()

    # Display results
    print("\nResults:")
    print(f"  Overall accuracy: {correct_total / 256:.2%}")
    print(f"  Left side accuracy: {correct_left / 256:.2%}")
    print(f"  Right side accuracy: {correct_right / 256:.2%}")
    print(f"  Final fitness: {winner.fitness:.2f}")

    # Analyze substrate structure
    print("\n" + "=" * 70)
    print("Substrate Network Analysis")
    print("=" * 70)
    print("\nConnection weight statistics:")
    print(
        f"  Input->Hidden weights: "
        f"mean={substrate_net.weights['input_hidden'].mean():.3f}, "
        f"std={substrate_net.weights['input_hidden'].std():.3f}"
    )
    print(
        f"  Hidden->Output weights: "
        f"mean={substrate_net.weights['hidden_output'].mean():.3f}, "
        f"std={substrate_net.weights['hidden_output'].std():.3f}"
    )


def main():
    """Main entry point."""
    import os

    # Create HyperNEAT config (CPPN has 6 inputs for source and target coords)
    config_content = """
[NEAT]
fitness_criterion     = max
fitness_threshold     = 999.0
pop_size              = 100
reset_on_extinction   = False
no_fitness_termination = False

[DefaultGenome]
# node activation options
activation_default      = sigmoid
activation_mutate_rate  = 0.2
activation_options      = sigmoid tanh

# node aggregation options
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

# node bias options
bias_init_type          = gaussian
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

# genome compatibility options
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

# connection add/remove rates
conn_add_prob           = 0.5
conn_delete_prob        = 0.5

# connection enable options
enabled_default         = True
enabled_mutate_rate     = 0.01
enabled_rate_to_true_add  = 0.0
enabled_rate_to_false_add = 0.0

feed_forward            = True
initial_connection      = full
single_structural_mutation = false
structural_mutation_surer = default

# node add/remove rates
node_add_prob           = 0.2
node_delete_prob        = 0.2

# network parameters (CPPN takes 6 inputs: x1,y1,z1,x2,y2,z2)
num_hidden              = 0
num_inputs              = 6
num_outputs             = 1

# node response options
response_init_type      = gaussian
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

# connection weight options
weight_init_type        = gaussian
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
min_species_size   = 2
"""

    # Write config file
    config_path = "hyperneat-config.ini"
    with open(config_path, "w") as f:
        f.write(config_content)

    print(f"Created HyperNEAT configuration: {config_path}")

    try:
        # Run HyperNEAT evolution
        winner, config = run_hyperneat(config_path, num_generations=30)

        # Test the winner
        test_winner(winner, config)

        print("\n" + "=" * 70)
        print("Evolution Complete!")
        print("=" * 70)

    finally:
        # Clean up
        if os.path.exists(config_path):
            os.remove(config_path)
            print(f"\nCleaned up: {config_path}")


if __name__ == "__main__":
    main()
