"""
This file contains different methods for selecting parents in a genetic algorithm.
Selection is the process of choosing which individuals from the current generation
will get to create offspring for the next generation. The idea is to give fitter
individuals a better chance to pass on their genes.
"""

import random
import numpy as np
from typing import List

class SelectionMethods:
    """
    A collection of static methods for parent selection in a genetic algorithm.
    These methods are used to pick out individuals from the population that will
    be used for crossover and mutation to create the next generation.
    """
    @staticmethod
    def tournament(population: List[np.ndarray],
                   fitnesses: List[float],
                   tournament_size: int = 3,
                   number_of_parents: int = 2) -> List[np.ndarray]:
        """
        Selects parents using a tournament.
        In a tournament, a small group of individuals is chosen at random,
        and the best one from that group is selected as a parent.
        This is repeated until we have enough parents.
        It's a simple and effective way to select for fitter individuals.
        """
        chosen = []
        for _ in range(number_of_parents):
            # Pick a few random individuals to compete
            contenders_indices = random.sample(range(len(population)), tournament_size)
            contender_fitnesses = [fitnesses[i] for i in contenders_indices]
            
            # The winner is the one with the lowest fitness (since we're minimizing)
            winner_index_in_contenders = np.argmin(contender_fitnesses)
            winner_index_in_population = contenders_indices[winner_index_in_contenders]
            
            chosen.append(population[winner_index_in_population].copy())
        return chosen

    @staticmethod
    def roulette_wheel(population: List[np.ndarray],
                       fitnesses: List[float],
                       number_of_parents: int = 2) -> List[np.ndarray]:
        """
        Selects parents using a roulette wheel, also known as fitness proportionate selection.
        Each individual is given a slice of a wheel that is proportional to its fitness.
        The wheel is spun, and wherever it lands, that individual is chosen.
        This gives every individual a chance, but fitter ones have a better chance.
        """
        # Since we are minimizing, we need to invert the fitness values so that
        # smaller fitness values (better solutions) have a larger slice of the wheel.
        max_fit = max(fitnesses)
        # We add a small epsilon to avoid division by zero if all fitnesses are the same.
        adjusted_fitnesses = [max_fit - f + 1e-9 for f in fitnesses]
        total_adjusted_fitness = sum(adjusted_fitnesses)
        probabilities = [f / total_adjusted_fitness for f in adjusted_fitnesses]

        chosen = []
        for _ in range(number_of_parents):
            # Spin the wheel
            spin = random.random()
            cumulative_prob = 0.0
            for i, prob in enumerate(probabilities):
                cumulative_prob += prob
                if spin <= cumulative_prob:
                    chosen.append(population[i].copy())
                    break
        return chosen