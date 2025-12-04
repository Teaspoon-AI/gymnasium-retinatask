"""NEAT evolution example for the Retina Task.

This script demonstrates how to evolve neural networks using NEAT (NeuroEvolution
of Augmenting Topologies) on the Retina Task.

Requires: neat-python
Install with: uv pip install neat-python
"""

import gymnasium as gym
import neat
import numpy as np

import gymnasium_retinatask  # noqa: F401


def eval_genome(genome, config):
    """Evaluate a NEAT genome on the Retina Task.

    Args:
        genome: NEAT genome to evaluate.
        config: NEAT configuration.

    Returns:
        Fitness score (higher is better).
    """
    # Create the network from the genome
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # Create environment in full evaluation mode
    env = gym.make("RetinaTask-v0", mode="full_evaluation", reward_type="paper")

    obs, info = env.reset()
    total_reward = 0.0

    while True:
        # NEAT network expects list input
        action = net.activate(obs.tolist())

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


def run_neat(config_path: str, num_generations: int = 100):
    """Run NEAT evolution.

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

    # Add reporters for monitoring progress
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    # Run evolution
    print("=" * 70)
    print("NEAT Evolution - Retina Task")
    print("=" * 70)
    print(f"\nEvolving for {num_generations} generations...")
    print("Each genome is evaluated on all 256 patterns.\n")

    winner = pop.run(eval_genomes, num_generations)

    # Display the winning genome
    print("\n" + "=" * 70)
    print("Best Genome")
    print("=" * 70)
    print(f"\nFitness: {winner.fitness:.2f}")
    print(f"Nodes: {len(winner.nodes)}")
    print(f"Connections: {len(winner.connections)}")

    return winner, config


def test_winner(winner, config):
    """Test the winning genome and display detailed results.

    Args:
        winner: Best genome.
        config: NEAT configuration.
    """
    print("\n" + "=" * 70)
    print("Testing Winner")
    print("=" * 70)

    # Create network
    net = neat.nn.FeedForwardNetwork.create(winner, config)

    # Test on environment
    env = gym.make("RetinaTask-v0", mode="full_evaluation")
    obs, info = env.reset()

    correct_total = 0
    correct_left = 0
    correct_right = 0
    patterns_evaluated = 0

    while True:
        # Get prediction
        action = net.activate(obs.tolist())

        obs, reward, terminated, truncated, info = env.step(np.array(action))

        # Check accuracy
        if info["pattern"] is not None:
            if info["pattern_error"] == 0.0:
                correct_total += 1
            if abs(action[0] - info["correct_left"]) < 0.5:
                correct_left += 1
            if abs(action[1] - info["correct_right"]) < 0.5:
                correct_right += 1

        patterns_evaluated += 1

        if terminated or truncated:
            break

    env.close()

    # Display results
    print("\nResults:")
    print(f"  Patterns evaluated: {patterns_evaluated}")
    print(f"  Overall accuracy: {correct_total / patterns_evaluated:.2%}")
    print(f"  Left side accuracy: {correct_left / patterns_evaluated:.2%}")
    print(f"  Right side accuracy: {correct_right / patterns_evaluated:.2%}")
    print(f"  Final fitness: {winner.fitness:.2f}")


def main():
    """Main entry point."""
    import os

    # Create NEAT config file
    config_content = """
[NEAT]
fitness_criterion     = max
fitness_threshold     = 999.0
pop_size              = 150
reset_on_extinction   = False
no_fitness_termination = False

[DefaultGenome]
# node activation options
activation_default      = sigmoid
activation_mutate_rate  = 0.0
activation_options      = sigmoid

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

# network parameters
num_hidden              = 0
num_inputs              = 8
num_outputs             = 2

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
    config_path = "neat-config.ini"
    with open(config_path, "w") as f:
        f.write(config_content)

    print(f"Created NEAT configuration: {config_path}")

    try:
        # Run NEAT evolution
        winner, config = run_neat(config_path, num_generations=50)

        # Test the winner
        test_winner(winner, config)

        print("\n" + "=" * 70)
        print("Evolution Complete!")
        print("=" * 70)

    finally:
        # Clean up config file
        if os.path.exists(config_path):
            os.remove(config_path)
            print(f"\nCleaned up: {config_path}")


if __name__ == "__main__":
    main()
