import os
import math
from collections import deque


def check_path(path):
    """Function for creating directory if it doesn't exist yet"""
    if not os.path.exists(path):
        os.mkdir(path)

def calculate_exponential_decay_rate(initial_value, final_value, total_iterations):
    """
    Calculates the exponential decay rate constant (k) for a given scenario.

    Args:
        initial_value (float): The starting value of the parameter (e.g., learning rate, epsilon).
        final_value (float): The desired final value of the parameter after total_iterations.
        total_iterations (int): The total number of iterations over which decay occurs.

    Returns:
        float: The exponential decay rate constant (k).
    """
    if initial_value <= 0 or final_value <= 0:
        raise ValueError("Initial and final values must be positive for exponential decay.")
    if total_iterations <= 0:
        raise ValueError("Total iterations must be a positive integer.")
    if final_value >= initial_value:
        raise ValueError("Final value must be less than the initial value for decay.")

    k = - (math.log(final_value / initial_value)) / total_iterations
    return k

def apply_exponential_decay(initial_value, final_value, decay_rate_k, current_iteration):
    """
    Applies exponential decay to a parameter based on the calculated decay rate.

    Args:
        initial_value (float): The starting value of the parameter.
        decay_rate_k (float): The exponential decay rate constant (k).
        current_iteration (int): The current iteration number.

    Returns:
        float: The decayed value of the parameter at the current iteration.
    """
    current_value = initial_value * math.exp(-decay_rate_k * current_iteration)
    return max(current_value, final_value)

class Buffer():
    """A general buffer class to store action indices or rewards."""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def __len__(self):
        return len(self.buffer)
    
    def append(self, action_index):
        self.buffer.append(action_index)
    
    def clear(self):
        self.buffer.clear()
    
    def test_all_same(self):
        return all(x == self.buffer[0] for x in self.buffer) if len(self.buffer) == self.buffer.maxlen else False

    def test_all_zero(self):
        return all(x == 0 for x in self.buffer) if len(self.buffer) == self.buffer.maxlen else False
    

    # def get_program_from_trees(self, trees_path):
    #     """Get a program representation of the policy learned by the set of trees."""
    #     # Get set of literals that partition the states
    #     self.load_trees(trees_path)
    #     literals = self.get_literals()
        
    #     if self.splits == "logical":
    #         # Get dimensions
    #         dimensions = []
    #         for literal in literals:
    #             dimension = literal.name.split("-")[-1]
    #             dimensions.append(dimension)
    #         dimensions = set(dimensions)

    #         # Get object pairs
    #         object_pairs = []
    #         for literal in literals:
    #             object_pair = (literal.obj1, literal.obj2)
    #             object_pairs.append(object_pair)
    #         object_pairs = set(object_pairs)
            
    #         # Get combionations of dimensions and object pair to group literals
    #         conditions = []
    #         for dimension in dimensions:
    #             for object_pair in object_pairs:
    #                 conditions.append((dimension, object_pair))
            
    #         dim_and_pairs = []
    #         for literal in literals:
    #             dimension = literal.name.split("-")[-1]
    #             object_pair = (literal.obj1, literal.obj2)
    #             dim_and_pairs.append((dimension, object_pair))

    #         # Group literals
    #         literal_types = []
    #         for condition in conditions:
    #             matching_literals = []
    #             for literal, dim_pair in zip(literals, dim_and_pairs):
    #                 if (dim_pair[0] == condition[0]) and (dim_pair[1] == condition[1]):
    #                     matching_literals.append(literal)
    #             if matching_literals:
    #                 literal_types.append(matching_literals)

    #         print(literal_types, len(literal_types))

    #         # Build states
    #         states = {}
    #         if len(literal_types) > 1:
    #             combinations = list(itertools.product(*literal_types))




    #         for literal_type in literal_types:
    #             for literal in literal_type:
    #                 combinations = list(itertools.product(list1, list2))
    #                 states.add(())


            

    #     return