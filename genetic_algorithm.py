"""
This module implements a Genetic Algorithm.
It's designed to be flexible, allowing for different types of problems (real-valued and binary),
and lets you swap out the key components of the algorithm like selection, crossover, and mutation.
Think of it as a toolkit for building your own evolutionary optimization solutions.
"""

import numpy as np
#import random
from typing import List, Dict, Any, Callable #Union

from objective_funcs import ObjectiveFunctions  # for type hints
from selection_methods import SelectionMethods
from crossover_operators import CrossoverOperators
from mutation_operators import MutationOperators

class GeneticAlgorithm:
    """
    A class that encapsulates the logic for a genetic algorithm.
    It's built to handle both real-valued and binary-encoded problems,
    making it adaptable for a wide range of optimization tasks.
    You can customize the selection, crossover, and mutation methods.
    """
    def __init__(
        self,
        objective_function: Callable[[np.ndarray], float],
        dimensions: int,
        bounds: tuple = (-5.0, 5.0),
        population_size: int = 50,
        max_generations: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.01,
        selection_method: str = 'tournament',
        crossover_operator: str = 'two_point',
        mutation_operator: str = 'gaussian',
        tournament_size: int = 3,
        representation: str = 'real', # can be switched to binary if needed
        bits_per_variable: int = 16
    ):
        """
        Sets up the genetic algorithm with all the necessary parameters.
        
        Args:
            objective_function: The function we want to minimize.
            dimensions: How many variables the objective function takes.
            bounds: The (min, max) range for each variable.
            population_size: How many individuals are in our population.
            max_generations: The maximum number of generations to run.
            crossover_rate: The chance that two parents will create offspring.
            mutation_rate: The chance that an individual will be mutated.
            selection_method: How we pick parents (e.g., 'tournament', 'roulette_wheel').
            crossover_operator: How we create children (e.g., 'two_point', 'uniform').
            mutation_operator: How we introduce random changes (e.g., 'gaussian', 'polynomial').
            tournament_size: If using tournament selection, this is the size of the tournament.
            representation: How we represent a solution, either 'real' numbers or 'binary' strings.
            bits_per_variable: For binary representation, how many bits to use for each variable.
        """
        # What we're trying to solve
        self.objective = objective_function
        self.dimensions = dimensions
        self.bounds = bounds

        # How we're representing a solution
        self.representation = representation
        self.bits_per_variable = bits_per_variable

        # The GA's settings
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.selection_method = selection_method
        self.crossover_operator = crossover_operator
        self.mutation_operator = mutation_operator
        self.tournament_size = tournament_size

        # To keep track of progress
        self.best_history: List[float] = []
        self.average_history: List[float] = []
        self.best_solution = None
        self.best_fitness = float('inf')

    def _decode_binary(self, genome: np.ndarray) -> np.ndarray:
        """
        Translates a binary genome into a real-valued vector.
        This is only used when the representation is 'binary'.
        It's like turning a string of 0s and 1s into a set of numbers that our objective function can understand.
        """
        vec = np.zeros(self.dimensions)
        lo, hi = self.bounds
        for i in range(self.dimensions):
            start = i * self.bits_per_variable
            block = genome[start:start + self.bits_per_variable]
            # Convert the binary block to an integer
            intval = int("".join(block.astype(str)), 2)
            # Scale the integer to fit within our bounds
            vec[i] = lo + (hi - lo) * intval / (2**self.bits_per_variable - 1)
        return vec

    def initialize_population(self) -> List[np.ndarray]:
        """
        Creates the first generation of individuals.
        If we're using real-valued representation, it's a list of random vectors.
        If we're using binary, it's a list of random bit arrays.
        """
        if self.representation == 'real':
            # Create individuals with random real values within the given bounds
            return [
                np.random.uniform(self.bounds[0], self.bounds[1], self.dimensions)
                for _ in range(self.population_size)
            ]
        # For binary, create individuals as random bit arrays
        total_bits = self.dimensions * self.bits_per_variable
        return [
            np.random.randint(0, 2, total_bits)
            for _ in range(self.population_size)
        ]

    def evaluate_population(self, population: List[np.ndarray]) -> List[float]:
        """
        Calculates the fitness for every individual in the population.
        Fitness is just the value of the objective function for that individual.
        It also keeps track of the best solution we've seen so far.
        """
        fitnesses = []
        for ind in population:
            # If it's a binary individual, we need to decode it first
            phen = self._decode_binary(ind) if self.representation == 'binary' else ind
            f = self.objective(phen)
            fitnesses.append(f)
            # Check if this is the best solution we've found yet
            if f < self.best_fitness:
                self.best_fitness = f
                self.best_solution = ind.copy()
        return fitnesses

    def select_parents(self, pop: List[np.ndarray], fits: List[float]) -> List[np.ndarray]:
        """
        Picks two parents from the population to create offspring.
        The method of selection (e.g., tournament, roulette wheel) is determined by the 'selection_method' parameter.
        """
        if self.selection_method == 'tournament':
            return SelectionMethods.tournament(pop, fits, self.tournament_size)
        return SelectionMethods.roulette_wheel(pop, fits)

    def crossover(self, p1: np.ndarray, p2: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Combines two parents to create two new children.
        This is where the "genetic" part of the algorithm really happens.
        The type of crossover is determined by the 'crossover_operator' parameter.
        """
        if self.representation == 'binary':
            # For binary, we always use one-point crossover for simplicity
            return CrossoverOperators.one_point(p1, p2, self.crossover_rate)
        if self.crossover_operator == 'uniform':
            return CrossoverOperators.uniform(p1, p2, self.crossover_rate)
        # Default to two-point crossover for real-valued representation
        return CrossoverOperators.two_point(p1, p2, self.crossover_rate)

    def mutate(self, ind: np.ndarray) -> np.ndarray:
        """
        Introduces a small, random change into an individual's genome.
        This helps to maintain diversity in the population and avoid getting stuck in local optima.
        The type of mutation is determined by the 'mutation_operator' parameter.
        """
        if self.representation == 'binary':
            # For binary, we flip bits
            return MutationOperators.flip_bit(ind, self.mutation_rate)
        if self.mutation_operator == 'polynomial':
            return MutationOperators.polynomial(ind, self.mutation_rate, bounds=self.bounds)
        # Default to Gaussian mutation for real-valued representation
        return MutationOperators.gaussian(ind, self.mutation_rate, bounds=self.bounds)

    def run(self) -> Dict[str, Any]:
        """
        The main engine of the genetic algorithm.
        It runs the evolutionary process for a set number of generations.
        In each generation, it evaluates, selects, crosses over, and mutates individuals
        to create a new population.
        """
        pop = self.initialize_population()
        for gen in range(self.max_generations):
            fits = self.evaluate_population(pop)
            # Record the best and average fitness for this generation
            self.best_history.append(min(fits))
            self.average_history.append(np.mean(fits))
            
            # If we've found a good enough solution, we can stop early
            if min(fits) < 1e-12:
                break

            # Elitism: Keep the top 2 individuals to ensure we don't lose the best solutions
            new_pop = [pop[i].copy() for i in np.argsort(fits)[:2]]

            # Create the rest of the new population through selection, crossover, and mutation
            while len(new_pop) < self.population_size:
                p1, p2 = self.select_parents(pop, fits)
                c1, c2 = self.crossover(p1, p2)
                new_pop.extend([self.mutate(c1), self.mutate(c2)])

            # The new population is ready for the next generation
            pop = new_pop[:self.population_size]

        # Return the results of the run
        return {
            'best_solution': self.best_solution,
            'best_fitness': self.best_fitness,
            'best_history': self.best_history,
            'average_history': self.average_history,
            'generations': len(self.best_history)
        }