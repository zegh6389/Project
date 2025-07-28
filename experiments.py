"""This script is for running experiments with the genetic algorithm.
It's set up to test different combinations of selection, crossover, and mutation
methods on a set of benchmark optimization problems. The goal is to see which
configurations work best for which types of problems.

The results, including convergence plots, are saved to disk."""

import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Dict, Any

import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for a cleaner output

# Import our custom modules
from objective_funcs import ObjectiveFunctions
from genetic_algorithm import GeneticAlgorithm


def plot_convergence(results: Dict[str, Dict[str, Any]]):
    """
    Creates and saves a plot showing the convergence of the genetic algorithm.
    For each problem, it plots the best and average fitness over generations
    for the best-performing configuration.
    """
    # Create a figure with 3 subplots, one for each problem
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (problem_name, problem_data) in zip(axes, results.items()):
        # Find the configuration that gave the best fitness for this problem
        best_config_key = min(problem_data, key=lambda k: problem_data[k]['best_fitness'])
        # Get the results from the first trial of that best configuration
        best_trial = problem_data[best_config_key]['trial_results'][0]
        generations = range(len(best_trial['best_history']))

        # Plot the best and average fitness, using a log scale for the y-axis
        # to better visualize improvements, especially when fitness values are small.
        ax.semilogy(generations, best_trial['best_history'], label='Best Fitness')
        ax.semilogy(
            generations,
            best_trial['average_history'],
            '--',
            label='Average Fitness',
            alpha=0.7
        )
        ax.set_title(f"{problem_name}\n(Best config: {best_config_key})")
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness (log scale)')
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.3)

    plt.tight_layout() # Adjusts plot to prevent labels from overlapping
    plt.savefig('convergence_analysis.png', dpi=300)  # Save the plot to a file
    plt.show()


def run_experiments():
    """
    Runs the main set of experiments.
    It systematically tests various GA configurations on different problems
    and collects all the results.
    """
    # Set random seeds to make our experiments reproducible.
    # This means we'll get the same results every time we run the script.
    np.random.seed(42)
    random.seed(42)

    # Define the problems we want to test our GA on.
    # Each problem has an objective function, bounds for the variables, and the number of dimensions.
    problems = {
        'De Jong Sphere': (
            ObjectiveFunctions.de_jong_sphere,
            (-5.12, 5.12),
            10
        ),
        'Rosenbrock': (
            ObjectiveFunctions.rosenbrock_valley,
            (-2.048, 2.048),
            2
        ),
        'Himmelblau': (
            ObjectiveFunctions.himmelblau_function,
            (-5.0, 5.0),
            2
        ),
    }

    # Define the different GA configurations we want to try.
    selections = ['tournament', 'roulette_wheel']
    operators = [
        ('two_point', 'gaussian'),
        ('uniform',   'gaussian'),
        ('two_point', 'polynomial'),
        ('uniform',   'polynomial'),
    ]
    tournament_sizes = [3, 5]
    num_trials = 10  # Number of times to repeat each experiment to get stable results

    all_results: Dict[str, Dict[str, Any]] = {}

    # Loop through each problem
    for name, (obj_func, bounds, dims) in problems.items():
        print(f"\n=== Testing Problem: {name} ===")
        results_for_this_problem: Dict[str, Any] = {}

        # Loop through each combination of selection, crossover, and mutation
        for selection_method in selections:
            # Tournament selection has an extra parameter (tournament size), so we test a few values.
            # For roulette wheel, we don't need a size, so we just use [None] to run the loop once.
            sizes_to_test = tournament_sizes if selection_method == 'tournament' else [None]
            for t_size in sizes_to_test:
                for crossover_op, mutation_op in operators:
                    config_label = f"{selection_method} | {crossover_op} | {mutation_op}"
                    if t_size:
                        config_label += f" (t_size={t_size})"
                    print(f"\nRunning configuration: {config_label}")
                    
                    fitness_scores = []
                    trial_outcomes = []

                    # Run each configuration multiple times to average out randomness
                    for i in range(1, num_trials + 1):
                        print(f"  - Trial {i}/{num_trials}...", end=' ')
                        ga = GeneticAlgorithm(
                            objective_function=obj_func,
                            dimensions=dims,
                            bounds=bounds,
                            population_size=50,
                            max_generations=150,
                            crossover_rate=0.8,
                            mutation_rate=0.02,
                            selection_method=selection_method,
                            crossover_operator=crossover_op,
                            mutation_operator=mutation_op,
                            tournament_size=t_size or 0,
                            representation='real'
                        )
                        run_output = ga.run()
                        fitness_scores.append(run_output['best_fitness'])
                        trial_outcomes.append(run_output)
                        print(f"Best fitness: {run_output['best_fitness']:.6f}")

                    # Calculate summary statistics for this configuration
                    mean_fitness = np.mean(fitness_scores)
                    std_dev_fitness = np.std(fitness_scores)
                    best_overall_fitness = np.min(fitness_scores)

                    print(f"  Summary: Best={best_overall_fitness:.6f}, Mean={mean_fitness:.6f} ± {std_dev_fitness:.6f}")

                    # Store the results for this configuration
                    config_key = f"{selection_method}_{crossover_op}_{mutation_op}"
                    if t_size: 
                        config_key += f"_t{t_size}"
                    results_for_this_problem[config_key] = {
                        'best_fitness': best_overall_fitness,
                        'mean_fitness': mean_fitness,
                        'std_fitness': std_dev_fitness,
                        'trial_results': trial_outcomes
                    }

        all_results[name] = results_for_this_problem

    print("\n--- Experiments Complete ---")
    print("Generating convergence plots...")
    plot_convergence(all_results)
    return all_results


if __name__ == "__main__":
    print("========== Starting Genetic Algorithm Experiments ==========")
    run_experiments()
    print("\nDone. Convergence plots have been saved to 'convergence_analysis.png'.")