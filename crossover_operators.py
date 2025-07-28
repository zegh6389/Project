"""
This file defines the crossover operators used in the genetic algorithm.
Crossover is how we combine two parent solutions to create new offspring.
It's inspired by the biological process of reproduction, where genetic material
from two parents is mixed to create a new, unique individual.
"""

import numpy as np
import random

class CrossoverOperators:
    """
    A collection of static methods for different types of crossover.
    These methods take two parents and produce two children, which will
    form part of the next generation in the genetic algorithm.
    """
    @staticmethod
    def one_point(parent1: np.ndarray,
                  parent2: np.ndarray,
                  rate: float = 0.8):
        """
        Performs one-point crossover.
        A single point is chosen in the parents' genomes, and the parts
        after that point are swapped. It's like cutting two strings at the
        same spot and then taping the opposite ends together.
        """
        # Check if we should perform crossover at all, based on the crossover rate
        if random.random() > rate:
            return parent1.copy(), parent2.copy()
        
        # Choose a random point to split the parents' genomes
        # We ensure the point is not at the very beginning or end
        split_point = random.randint(1, len(parent1) - 1)
        
        # Create the children by swapping the tails of the parents
        child1 = np.concatenate([parent1[:split_point], parent2[split_point:]])
        child2 = np.concatenate([parent2[:split_point], parent1[split_point:]])
        
        return child1, child2

    @staticmethod
    def two_point(parent1: np.ndarray,
                  parent2: np.ndarray,
                  rate: float = 0.8):
        """
        Performs two-point crossover.
        Two points are chosen, and the segment between these points is swapped
        between the parents. This is useful for preserving "schemas" or blocks
        of genes that are close together.
        """
        # Check if we should perform crossover
        if random.random() > rate:
            return parent1.copy(), parent2.copy()
        
        num_genes = len(parent1)
        # If the parents are too short, two-point crossover doesn't make sense
        if num_genes < 3:
            return CrossoverOperators.one_point(parent1, parent2, rate)
            
        # Choose two distinct points to cut
        point1, point2 = sorted(random.sample(range(1, num_genes), 2))
        
        # Create children by swapping the middle segments
        child1 = np.concatenate([parent1[:point1], parent2[point1:point2], parent1[point2:]])
        child2 = np.concatenate([parent2[:point1], parent1[point1:point2], parent2[point2:]])
        
        return child1, child2

    @staticmethod
    def uniform(parent1: np.ndarray,
                parent2: np.ndarray,
                rate: float = 0.8):
        """
        Performs uniform crossover.
        For each gene, we flip a coin to decide which parent it should come from.
        This allows for a lot of mixing and can be good for exploring the search space,
        but it might break up good combinations of genes.
        """
        # Check if we should perform crossover
        if random.random() > rate:
            return parent1.copy(), parent2.copy()
        
        # Create a "mask" of random booleans.
        # If the mask is True, the gene comes from parent2. Otherwise, from parent1.
        mask = np.random.rand(len(parent1)) < 0.5
        
        # np.where is a handy way to apply this mask
        child1 = np.where(mask, parent2, parent1)
        child2 = np.where(mask, parent1, parent2) # The second child is the inverse of the first
        
        return child1, child2