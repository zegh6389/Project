"""
This file defines a few common objective functions used for testing optimization algorithms.
Objective functions, also known as cost functions or fitness functions, are what we're trying to minimize.
Each function has a different shape and complexity, which helps in testing how well an algorithm
can navigate different types of search spaces
"""

import numpy as np

class ObjectiveFunctions:
    """
    A collection of static methods, each representing a different objective function.
    These are classic test functions in the field of optimization
    """
    @staticmethod
    def de_jong_sphere(vector: np.ndarray) -> float:
        """
        The De Jong's sphere function.
        It's a simple, convex function, shaped like a bowl. The minimum is at (0, 0, ..., 0).
        It's a good first test to see if an algorithm can find a simple minimum
        """
        return float(np.sum(vector**2))

    @staticmethod
    def rosenbrock_valley(vector: np.ndarray) -> float:
        """
        The Rosenbrock valley function, also known as the banana function.
        It has a narrow, parabolic valley. It's tricky because while the valley is easy to find,
        converging to the actual minimum is difficult. It's a classic test of an algorithm's
        ability to handle non-convex, ill-conditioned problems
        """
        if vector.size < 2:
            raise ValueError("The Rosenbrock function needs at least two dimensions.")
        # The formula is a sum over all dimensions
        return float(np.sum(100.0 * (vector[1:] - vector[:-1]**2)**2 + (1.0 - vector[:-1])**2))

    @staticmethod
    def himmelblau_function(vector: np.ndarray) -> float:
        """
        Himmelblau's function.
        This function is interesting because it has four identical minima
        It's a good test to see if an algorithm can find multiple solutions, or if it just
        converges to the first one it finds
        It's only defined for two dimensions 
        """
        if vector.size != 2:
            raise ValueError("Himmelblau's function is only defined for two dimensions.")
        x, y = vector
        # The classic formula for Himmelblau's function
        return float((x**2 + y - 11.0)**2 + (x + y**2 - 7.0)**2)