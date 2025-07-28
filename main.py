"""This is the main entry point for running the genetic algorithm experiments.
When you run this file, it kicks off the whole process defined in `experiments.py`.
It's kept simple on purpose, so you can easily see where the execution starts.

"""
from experiments import run_experiments

if __name__ == "__main__":
    # A friendly message to let you know the experiments are starting.
    print("=== Kicking off Genetic Algorithm Experiments ===")
    
    # This function does all the heavy lifting.
    run_experiments()
    
    # And a final message to confirm everything is done.
    print("\nAll experiments are complete. Check for 'convergence_analysis.png' to see the results.")