"""
This file provides mutation operators for the genetic algorithm.
Mutation is a crucial part of the GA, as it introduces random changes into the population.
This helps to maintain genetic diversity and prevents the algorithm from getting stuck
in a local optimum. It's the main way we explore new parts of the search space.
"""

import numpy as np
import random

class MutationOperators:
    """
    A collection of static methods for different types of mutation.
    Each method takes an individual's genome and a mutation rate, and returns
    a potentially modified genome.
    """
    @staticmethod
    def flip_bit(genome: np.ndarray, rate: float = 0.01) -> np.ndarray:
        """
        Performs bit-flip mutation, for binary-encoded genomes.
        It goes through each bit in the genome and has a small chance (the mutation rate)
        of flipping it from 0 to 1 or 1 to 0.
        """
        mutated_genome = genome.copy()
        for i in range(len(mutated_genome)):
            if random.random() < rate:
                # Flip the bit
                mutated_genome[i] = 1 - mutated_genome[i]
        return mutated_genome

    @staticmethod
    def gaussian(genome: np.ndarray,
                 rate: float = 0.01,
                 sigma: float = 0.1,
                 bounds: tuple = (-5.0, 5.0)) -> np.ndarray:
        """
        Applies Gaussian mutation to a real-valued genome.
        For each gene, it has a small chance of adding a random value drawn from a
        Gaussian (normal) distribution. This is like giving the gene a small nudge.
        The `sigma` parameter controls the size of that nudge.
        """
        mutated_genome = genome.copy()
        lower_bound, upper_bound = bounds
        for i in range(len(mutated_genome)):
            if random.random() < rate:
                # Add a small random value from a normal distribution
                noise = np.random.normal(0, sigma)
                mutated_genome[i] += noise
                # Make sure the new value is still within the allowed bounds
                mutated_genome[i] = np.clip(mutated_genome[i], lower_bound, upper_bound)
        return mutated_genome

    @staticmethod
    def polynomial(genome: np.ndarray,
                   rate: float = 0.01,
                   eta: float = 20.0,
                   bounds: tuple = (-5.0, 5.0)) -> np.ndarray:
        """
        Applies polynomial mutation, a more sophisticated method for real-valued genomes.
        It creates a perturbation that is more likely to be small than large, and it's
        aware of the variable's bounds, so it's less likely to make big jumps when
        near the edge of the search space. The `eta` parameter controls the shape of the
        probability distribution.
        """
        mutated_genome = genome.copy()
        lower_bound, upper_bound = bounds
        for i in range(len(mutated_genome)):
            if random.random() < rate:
                y = mutated_genome[i]
                # Calculate how close the value is to the bounds
                dist_to_lower = (y - lower_bound) / (upper_bound - lower_bound)
                dist_to_upper = (upper_bound - y) / (upper_bound - lower_bound)
                
                rand_val = random.random()
                power = 1.0 / (eta + 1.0)

                # This formula calculates the perturbation, giving a higher chance
                # to smaller changes.
                if rand_val < 0.5:
                    xy = 1.0 - dist_to_lower
                    val = 2.0 * rand_val + (1.0 - 2.0 * rand_val) * (xy ** (eta + 1.0))
                    delta = val ** power - 1.0
                else:
                    xy = 1.0 - dist_to_upper
                    val = 2.0 * (1.0 - rand_val) + 2.0 * (rand_val - 0.5) * (xy ** (eta + 1.0))
                    delta = 1.0 - (val ** power)
                
                # Apply the change and ensure it stays within bounds
                new_value = y + delta * (upper_bound - lower_bound)
                mutated_genome[i] = np.clip(new_value, lower_bound, upper_bound)
        return mutated_genome