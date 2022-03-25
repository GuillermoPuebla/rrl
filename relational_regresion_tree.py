import os
import math
import itertools
import pickle
import random
import csv
from PIL import Image
from collections import deque, namedtuple
import numpy as np
import pandas as pd
from scipy.stats import f
import graphviz
import gym
from preprocessing import BreakoutPreprocessor, PongPreprocessor, DemonAttackPreprocessor, \
    get_state_comparative_breakout, get_state_logical_breakout, \
        get_state_comparative_pong, get_state_logical_pong, \
            get_state_comparative_demon_attack, get_state_logical_demon_attack, Fact

# Method for creating directory if it doesn't exist yet
def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

class Buffer():
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


# Literal class. The last two arguments are optional
Literal = namedtuple('Literal', ['name', 'type', 'obj1', 'obj2', 'obj3'], defaults=[None, None])

class Node():
    """A RelationalRegressionNode based on more/same/less partitions."""
    def __init__(
        self,
        root_action,
        refinements=None,
        max_depth=None,
        min_sample_size=None,
        depth=None,
        node_type=None,
        parent=None,
        literal=None,
        conjuntion_str=None,
        significance_level=None,
        across_objects_bias=None,
        consisten_dim_bias=None,
        best_literal_criteria=None,
        valid_actions=None
        ):
        # Always require a root action.
        self.root_action = root_action
        # Define node possible refinements.
        self.refinements = refinements if refinements else []
        # Saving the hyper parameters.
        self.max_depth = max_depth if max_depth else 9
        self.min_sample_size = min_sample_size if min_sample_size else 10000
        # Default current depth of node.
        self.depth = depth if depth else 0
        # Type of node.
        self.node_type = node_type if node_type else 'root'
        # Node parent.
        self.parent = parent if parent else None
        # Literal that splits the node.
        self.literal = literal if literal else None
        # Initialize q-value. Used to make and update the node predictions.
        self.q_value = 0.0
        # Initialize statistics dictionary.
        self.refinements_stats = self.make_stats_dict()
        # Always initialize the more, same and less braches as None
        self.more_branch = None 
        self.same_branch = None
        self.less_branch = None
        self.children = []
        # Get node string identifier for graphviz
        self.conjuntion_str = conjuntion_str if conjuntion_str else self.root_action + 'root'
        # Set significance level for f-test.
        self.significance_level = significance_level if significance_level else 0.01
        # Set biases.
        self.across_objects_bias = across_objects_bias if across_objects_bias else False
        self.consisten_dim_bias = consisten_dim_bias if consisten_dim_bias else False
        # Best literal criteria
        self.best_literal_criteria = best_literal_criteria if best_literal_criteria else 'f-ratio'
        # Action to consider when spliting
        self.valid_actions = valid_actions if valid_actions else []
    def make_stats_dict(self):
        stats = {}
        for literal in self.refinements:
            # Comparative literal example: ('x', 'paddle', 'ball')
            if literal.type == 'comparative':
                literals_dict = {
                    literal: {
                        'Total': {'n': 0, 'mu': 0, 'S': 0},
                        'same': {'n': 0, 'mu': 0, 'S': 0},
                        'more': {'n': 0, 'mu': 0, 'S': 0},
                        'less': {'n': 0, 'mu': 0, 'S': 0}
                        }}
            # Logical literal example: ('more-x', 'paddle', 'ball')
            elif literal.type == 'logical':
                literals_dict = {
                    literal: {
                        'Total': {'n': 0, 'mu': 0, 'S': 0},
                        'True': {'n': 0, 'mu': 0, 'S': 0},
                        'False': {'n': 0, 'mu': 0, 'S': 0}
                    }}
            else:
                raise ValueError("Unrecognized literal type!")
            stats.update(literals_dict)
        return stats

    def get_f_ratio(self, literal):
        """Returns the ratio of the variance of the q-values of the examples
        collected in a leaf before and after splitting."""
        # Because S = sigma * n, dividing S by n_{Total} gives the weighted variance.
        try:
            weighted_sigma_same = self.refinements_stats[literal]['same']['S'] / self.refinements_stats[literal]['Total']['n']
            weighted_sigma_more = self.refinements_stats[literal]['more']['S'] / self.refinements_stats[literal]['Total']['n']
            weighted_sigma_less = self.refinements_stats[literal]['less']['S'] / self.refinements_stats[literal]['Total']['n']
            sigma_overall = self.refinements_stats[literal]['Total']['S'] / self.refinements_stats[literal]['Total']['n']
            return (weighted_sigma_same + weighted_sigma_more + weighted_sigma_less) / sigma_overall
        except KeyError:
            weighted_sigma_true = self.refinements_stats[literal]['True']['S'] / self.refinements_stats[literal]['Total']['n']
            weighted_sigma_false = self.refinements_stats[literal]['False']['S'] / self.refinements_stats[literal]['Total']['n']
            sigma_overall = self.refinements_stats[literal]['Total']['S'] / self.refinements_stats[literal]['Total']['n']
            return (weighted_sigma_true + weighted_sigma_false) / sigma_overall

    def get_max_q_value(self, literal):
        try:
            q_same = self.refinements_stats[literal]['same']['mu']
            q_more = self.refinements_stats[literal]['more']['mu']
            q_less = self.refinements_stats[literal]['less']['mu']
            q_vals = [q_same, q_more, q_less]
        except KeyError:
            q_true = self.refinements_stats[literal]['True']['mu']
            q_false = self.refinements_stats[literal]['False']['mu']
            q_vals = [q_true, q_false]

        return max(q_vals)

    def find_best_literal(self):
        """
        Find the best literal to split the node by iterating over each
        possible refinement and calculating the f-ratio. If there are
        no literals with enough samples, returns None.
        """
        
        best_literal = None
        best_f_ratio = 1.0
        best_q_value = -np.inf
        best_p_value = 1.0
        
        # Select literals that have reached the minimum number of samples.
        valid_literals = [x for x in self.refinements if self.refinements_stats[x]['Total']['n'] > self.min_sample_size]

        # When the across objects bias is active the only valid literals for the root nodes are between-objects relations.
        if self.across_objects_bias:
            if self.node_type == 'root':
                valid_literals = [x for x in valid_literals if x.obj2.endswith('t')]

        if self.consisten_dim_bias:            
            # For all children nodes the valid literals have the same dimmension as the parent node.
            if self.node_type != 'root':
                valid_literals = [x for x in valid_literals if x.name == self.parent.literal.name]
        
        if valid_literals:
            for literal in valid_literals:
                if self.best_literal_criteria == 'max-q-value':
                    q_value = self.get_max_q_value(literal)
                    if q_value >= best_q_value:
                        best_literal = literal
                        best_q_value = q_value
                elif self.best_literal_criteria == 'f-ratio':
                    f_ratio = self.get_f_ratio(literal)
                    if f_ratio <= best_f_ratio:
                        best_literal = literal
                        best_f_ratio = f_ratio
                elif self.best_literal_criteria == 'p-value':
                    f_ratio = self.get_f_ratio(literal)
                    n = self.refinements_stats[literal]['Total']['n']
                    p_value = f.cdf(f_ratio, n-1, n-1) # One-tailed "less" F-test
                    if p_value <= best_p_value:
                        best_literal = literal
                        best_p_value = p_value
                else:
                    raise ValueError('Unrecognized best literal criteria!')
        return best_literal

    def f_test(self, f_ratio, n, m):
        """Runs a F-test of equality of variances."""
        df1 = n - 1
        df1 = m - 1
        # One-tailed "less" F-test
        p_value = f.cdf(f_ratio, df1, df1)
        return p_value <= self.significance_level

    def predict(self, state):
        """
        Predicts a q-value for a state by traversing the tree until
        the last branch.
        This method should be called from a root node only.
        """
        # Initialize current node.
        current_node = self

        # Base case: we've reached a leaf.
        if current_node.literal is None:
            return current_node.q_value

        # Base case 2: state does not have any facts.
        if not state:
            return current_node.q_value
        
        # Traversing the nodes all the way to the bottom.
        while current_node.literal is not None:
            # Get relevant literal.
            literal =  current_node.literal
            fact = [x for x in state if x.name == literal.name and x.obj1 == literal.obj1 and x.obj2 == literal.obj2]
            if fact:
                fact = fact[0]
            else:
                break
            if fact.value == 'more':
                current_node = current_node.more_branch
                current_node.predict(state)
            elif fact.value == 'same':
                current_node = current_node.same_branch
                current_node.predict(state)
            elif fact.value == 'less':
                current_node = current_node.less_branch
                current_node.predict(state)
            elif fact.value == 'True':
                current_node = current_node.true_branch
                current_node.predict(state)
            elif fact.value == 'False':
                current_node = current_node.false_branch
                current_node.predict(state)
            else:
                print('state_literal: ', fact)
                print('state_literal[1]: ', fact.value)
                raise ValueError('Unrecognised comparative!')
        
        return current_node.q_value

    def get_state_leaf(self, state):
        """Returns the most specific leaf node corresponding to the state."""

        # Initialize current node.
        current_node = self

        # Base case: we've reached a leaf.
        if current_node.literal is None:
            return current_node

        # Base case 2: state does not have any facts.
        if not state:
            return current_node
        
        # Traversing the nodes all the way to the bottom.
        while current_node.literal is not None:
            # Get relevant literal.
            literal =  current_node.literal
            fact = [x for x in state if x.name == literal.name and x.obj1 == literal.obj1 and x.obj2 == literal.obj2]
            if fact:
                fact = fact[0]
            else:
                break
            if fact.value == 'more':
                current_node = current_node.more_branch
                current_node.get_state_leaf(state)
            elif fact.value == 'same':
                current_node = current_node.same_branch
                current_node.get_state_leaf(state)
            elif fact.value == 'less':
                current_node = current_node.less_branch
                current_node.get_state_leaf(state)
            elif fact.value == 'True':
                current_node = current_node.true_branch
                current_node.predict(state)
            elif fact.value == 'False':
                current_node = current_node.false_branch
                current_node.predict(state)
            else:
                raise ValueError('Unrecognised comparative!')
        
        return current_node

    def update_statistics_old(self, state):
        """
        Update the statistics of the leaf node corresponding to the state.
        This method should be called from a root node only.
        Update equations:
            n_t = n_{t-1} + 1
            mu_t = mu_{t-1} + (q_t - mu_{t-1})/n_t
            S_t = S_{t-1} + (q_t - mu_{t-1}) * (q_t - mu_t)
        """
        # Never update the 'FIRE' root.
        # if self.root_action == 'FIRE':
        #     return
        # If there are no relations in the state return inmediatly.
        facts = [x for x in state if x.obj2 is not None]
        if not facts:
            return
        # Traverse the nodes all the way to the bottom.
        current_node = self.get_state_leaf(state)
        for literal in current_node.refinements:
            # Update the literal only if objects are present in the state.
            object_1 = [x for x in state if x.name == 'present' and x.value == 'True' and x.obj1 == literal.obj1]
            object_2 = [x for x in state if x.name == 'present' and x.value == 'True' and x.obj1 == literal.obj2]
            if object_1 and object_2:
                # Always update total.
                # Update n.
                current_node.refinements_stats[literal]['Total']['n'] += 1
                # Update mean.
                n = current_node.refinements_stats[literal]['Total']['n']
                old_mean = current_node.refinements_stats[literal]['Total']['mu']
                current_node.refinements_stats[literal]['Total']['mu'] = old_mean + (current_node.q_value - old_mean) / n
                # Update S (variance times n).
                new_mean = current_node.refinements_stats[literal]['Total']['mu']
                old_s = current_node.refinements_stats[literal]['Total']['S']
                current_node.refinements_stats[literal]['Total']['S'] = \
                    old_s + (current_node.q_value - old_mean) * (current_node.q_value - new_mean)
                # The relevant state literal matches the node refinement.
                facts = [x for x in state if x.obj2 is not None]
                fact = [x for x in facts if x.name == literal.name and x.obj1 == literal.obj1 and x.obj2 == literal.obj2][0]
                if fact.value == 'same':
                    # Update n.
                    current_node.refinements_stats[literal]['same']['n'] += 1
                    # Update mean.
                    n = current_node.refinements_stats[literal]['same']['n']
                    old_mean = current_node.refinements_stats[literal]['same']['mu']
                    current_node.refinements_stats[literal]['same']['mu'] = old_mean + (current_node.q_value - old_mean) / n
                    # Update S (variance times n).
                    new_mean = current_node.refinements_stats[literal]['same']['mu']
                    old_s = current_node.refinements_stats[literal]['same']['S']
                    current_node.refinements_stats[literal]['same']['S'] = \
                        old_s + (current_node.q_value - old_mean) * (current_node.q_value - new_mean)
                elif fact.value == 'more':
                    # Update n.
                    current_node.refinements_stats[literal]['more']['n'] += 1
                    # Update mean.
                    n = current_node.refinements_stats[literal]['more']['n']
                    old_mean = current_node.refinements_stats[literal]['more']['mu']
                    current_node.refinements_stats[literal]['more']['mu'] = old_mean + (current_node.q_value - old_mean) / n
                    # Update S (variance times n).
                    new_mean = current_node.refinements_stats[literal]['more']['mu']
                    old_s = current_node.refinements_stats[literal]['more']['S']
                    current_node.refinements_stats[literal]['more']['S'] = \
                        old_s + (current_node.q_value - old_mean) * (current_node.q_value - new_mean)
                elif fact.value == 'less':
                    # Update n.
                    current_node.refinements_stats[literal]['less']['n'] += 1
                    # Update mean.
                    n = current_node.refinements_stats[literal]['less']['n']
                    old_mean = current_node.refinements_stats[literal]['less']['mu']
                    current_node.refinements_stats[literal]['less']['mu'] = old_mean + (current_node.q_value - old_mean) / n
                    # Update S (variance times n).
                    new_mean = current_node.refinements_stats[literal]['less']['mu']
                    old_s = current_node.refinements_stats[literal]['less']['S']
                    current_node.refinements_stats[literal]['less']['S'] = \
                        old_s + (current_node.q_value - old_mean) * (current_node.q_value - new_mean)
                elif fact.value == 'True':
                    # Update n.
                    current_node.refinements_stats[literal]['True']['n'] += 1
                    # Update mean.
                    n = current_node.refinements_stats[literal]['True']['n']
                    old_mean = current_node.refinements_stats[literal]['True']['mu']
                    current_node.refinements_stats[literal]['True']['mu'] = old_mean + (current_node.q_value - old_mean) / n
                    # Update S (variance times n).
                    new_mean = current_node.refinements_stats[literal]['True']['mu']
                    old_s = current_node.refinements_stats[literal]['True']['S']
                    current_node.refinements_stats[literal]['True']['S'] = \
                        old_s + (current_node.q_value - old_mean) * (current_node.q_value - new_mean)
                elif fact.value == 'False':
                    # Update n.
                    current_node.refinements_stats[literal]['False']['n'] += 1
                    # Update mean.
                    n = current_node.refinements_stats[literal]['False']['n']
                    old_mean = current_node.refinements_stats[literal]['False']['mu']
                    current_node.refinements_stats[literal]['False']['mu'] = old_mean + (current_node.q_value - old_mean) / n
                    # Update S (variance times n).
                    new_mean = current_node.refinements_stats[literal]['False']['mu']
                    old_s = current_node.refinements_stats[literal]['False']['S']
                    current_node.refinements_stats[literal]['False']['S'] = \
                        old_s + (current_node.q_value - old_mean) * (current_node.q_value - new_mean)
                else:
                    raise ValueError('Unrecognised comparative value!')
        return
    
    def update_statistics(self, state):
        """
        Update the statistics of the leaf node corresponding to the state.
        This method should be called from a root node only.
        Update equations:
            n_t = n_{t-1} + 1
            mu_t = mu_{t-1} + (q_t - mu_{t-1})/n_t
            S_t = S_{t-1} + (q_t - mu_{t-1}) * (q_t - mu_t)
        """
        # Never update the 'FIRE' root.
        # if self.root_action == 'FIRE':
        #     return
        # If there are no relations in the state return inmediatly.
        if not state:
            return
        # Traverse the nodes all the way to the bottom.
        current_node = self.get_state_leaf(state)
        for literal in current_node.refinements:
            relevant_facts = [x for x in state if x.name == literal.name and x.obj1 == literal.obj1 and x.obj2 == literal.obj2]
            if relevant_facts:
                # Always update total.
                # Update n.
                current_node.refinements_stats[literal]['Total']['n'] += 1
                # Update mean.
                n = current_node.refinements_stats[literal]['Total']['n']
                old_mean = current_node.refinements_stats[literal]['Total']['mu']
                current_node.refinements_stats[literal]['Total']['mu'] = old_mean + (current_node.q_value - old_mean) / n
                # Update S (variance times n).
                new_mean = current_node.refinements_stats[literal]['Total']['mu']
                old_s = current_node.refinements_stats[literal]['Total']['S']
                current_node.refinements_stats[literal]['Total']['S'] = \
                    old_s + (current_node.q_value - old_mean) * (current_node.q_value - new_mean)
                
                for fact in relevant_facts:
                    if fact.value == 'same':
                        # Update n.
                        current_node.refinements_stats[literal]['same']['n'] += 1
                        # Update mean.
                        n = current_node.refinements_stats[literal]['same']['n']
                        old_mean = current_node.refinements_stats[literal]['same']['mu']
                        current_node.refinements_stats[literal]['same']['mu'] = old_mean + (current_node.q_value - old_mean) / n
                        # Update S (variance times n).
                        new_mean = current_node.refinements_stats[literal]['same']['mu']
                        old_s = current_node.refinements_stats[literal]['same']['S']
                        current_node.refinements_stats[literal]['same']['S'] = \
                            old_s + (current_node.q_value - old_mean) * (current_node.q_value - new_mean)
                    elif fact.value == 'more':
                        # Update n.
                        current_node.refinements_stats[literal]['more']['n'] += 1
                        # Update mean.
                        n = current_node.refinements_stats[literal]['more']['n']
                        old_mean = current_node.refinements_stats[literal]['more']['mu']
                        current_node.refinements_stats[literal]['more']['mu'] = old_mean + (current_node.q_value - old_mean) / n
                        # Update S (variance times n).
                        new_mean = current_node.refinements_stats[literal]['more']['mu']
                        old_s = current_node.refinements_stats[literal]['more']['S']
                        current_node.refinements_stats[literal]['more']['S'] = \
                            old_s + (current_node.q_value - old_mean) * (current_node.q_value - new_mean)
                    elif fact.value == 'less':
                        # Update n.
                        current_node.refinements_stats[literal]['less']['n'] += 1
                        # Update mean.
                        n = current_node.refinements_stats[literal]['less']['n']
                        old_mean = current_node.refinements_stats[literal]['less']['mu']
                        current_node.refinements_stats[literal]['less']['mu'] = old_mean + (current_node.q_value - old_mean) / n
                        # Update S (variance times n).
                        new_mean = current_node.refinements_stats[literal]['less']['mu']
                        old_s = current_node.refinements_stats[literal]['less']['S']
                        current_node.refinements_stats[literal]['less']['S'] = \
                            old_s + (current_node.q_value - old_mean) * (current_node.q_value - new_mean)
                    elif fact.value == 'True':
                        # Update n.
                        current_node.refinements_stats[literal]['True']['n'] += 1
                        # Update mean.
                        n = current_node.refinements_stats[literal]['True']['n']
                        old_mean = current_node.refinements_stats[literal]['True']['mu']
                        current_node.refinements_stats[literal]['True']['mu'] = old_mean + (current_node.q_value - old_mean) / n
                        # Update S (variance times n).
                        new_mean = current_node.refinements_stats[literal]['True']['mu']
                        old_s = current_node.refinements_stats[literal]['True']['S']
                        current_node.refinements_stats[literal]['True']['S'] = \
                            old_s + (current_node.q_value - old_mean) * (current_node.q_value - new_mean)
                    elif fact.value == 'False':
                        # Update n.
                        current_node.refinements_stats[literal]['False']['n'] += 1
                        # Update mean.
                        n = current_node.refinements_stats[literal]['False']['n']
                        old_mean = current_node.refinements_stats[literal]['False']['mu']
                        current_node.refinements_stats[literal]['False']['mu'] = old_mean + (current_node.q_value - old_mean) / n
                        # Update S (variance times n).
                        new_mean = current_node.refinements_stats[literal]['False']['mu']
                        old_s = current_node.refinements_stats[literal]['False']['S']
                        current_node.refinements_stats[literal]['False']['S'] = \
                            old_s + (current_node.q_value - old_mean) * (current_node.q_value - new_mean)
                    else:
                        raise ValueError('Unrecognised comparative value!')
             
        return

    def split_iteration(self, state, inherit_q_values):
        """Splits the leaf node corresponding to the state if the
        best literal passes the F-test."""

        # If the root action is 'FIRE' don't split.
        # if self.root_action == 'FIRE' or self.root_action == 'RIGHTFIRE' or self.root_action == 'LEFTFIRE':
        #     return False
        # If the root action is not part of the valid actions don't split
        if self.root_action not in self.valid_actions:
            return False
        
        # If there are no between-object relations in the state return inmediatly.
        # relations = [x for x in state if len(x) == 4 and x[2].endswith('0')]
        # if not relations:
        #     return False
        facts = [x for x in state if x.obj2 is not None]
        if not facts:
            return False
        
        # Find relevant node.
        node = self.get_state_leaf(state)

        # If node depth is equal or bigger than max_depth return inmediatly.
        if node.depth >= self.max_depth:
            return False

        # If the node is not a leaf return inmediatly.
        if node.literal is not None:
            return False
        
        # Find the best literal and F-value.
        best_literal = node.find_best_literal()

        # If none of the relations has reached the minimun number of samples return inmediatly.
        if best_literal is None:
            return False

        # Run F-test.
        # n is the overall sample size.
        n = node.refinements_stats[best_literal]['Total']['n']
        best_f_ratio = node.get_f_ratio(best_literal)
        f_test = self.f_test(best_f_ratio, n=n, m=n)

        # Split the node if f-test is postive.
        if f_test:
            children_refinements = [x for x in node.refinements if x != best_literal]
            # Literal = namedtuple('Literal', ['name', 'type', 'obj1', 'obj2', 'obj3'], defaults=[None, None])
            if best_literal.type == 'comparative':
                more_branch = Node(
                    root_action=node.root_action,
                    refinements=children_refinements,
                    max_depth=node.max_depth,
                    min_sample_size=node.min_sample_size,
                    node_type='more',
                    parent=node,
                    depth=node.depth+1,
                    conjuntion_str=node.conjuntion_str + str(best_literal) + 'more',
                    significance_level=node.significance_level,
                    across_objects_bias=node.across_objects_bias,
                    consisten_dim_bias=node.consisten_dim_bias,
                    best_literal_criteria=node.best_literal_criteria,
                    valid_actions=node.valid_actions
                    )
                same_branch = Node(
                    root_action=node.root_action,
                    refinements=children_refinements,
                    max_depth=node.max_depth,
                    min_sample_size=node.min_sample_size,
                    node_type='same',
                    parent=node,
                    depth=node.depth+1,
                    conjuntion_str=node.conjuntion_str + str(best_literal) + 'same',
                    significance_level=node.significance_level,
                    across_objects_bias=node.across_objects_bias,
                    consisten_dim_bias=node.consisten_dim_bias,
                    best_literal_criteria=node.best_literal_criteria,
                    valid_actions=node.valid_actions
                    )
                less_branch = Node(
                    root_action=node.root_action,
                    refinements=children_refinements,
                    max_depth=node.max_depth,
                    min_sample_size=node.min_sample_size,
                    node_type='less',
                    parent=node,
                    depth=node.depth+1,
                    conjuntion_str=node.conjuntion_str + str(best_literal) + 'less',
                    significance_level=node.significance_level,
                    across_objects_bias=node.across_objects_bias,
                    consisten_dim_bias=node.consisten_dim_bias,
                    best_literal_criteria=node.best_literal_criteria,
                    valid_actions=node.valid_actions
                    )
                # Inherit q-values from parent node.
                if inherit_q_values:
                    more_branch.q_value = node.q_value #node.refinements_stats[best_literal]['more']['mu']
                    same_branch.q_value = node.q_value #node.refinements_stats[best_literal]['same']['mu']
                    less_branch.q_value = node.q_value #node.refinements_stats[best_literal]['less']['mu']

                # Make reference to children in parent node.
                node.more_branch = more_branch
                node.same_branch = same_branch
                node.less_branch = less_branch
                node.children = [node.more_branch, node.same_branch, node.less_branch]
            # Literal = namedtuple('Literal', ['name', 'type', 'obj1', 'obj2', 'obj3'], defaults=[None, None])
            elif best_literal.type == 'logical':
                true_branch = Node(
                    root_action=node.root_action,
                    refinements=children_refinements,
                    max_depth=node.max_depth,
                    min_sample_size=node.min_sample_size,
                    node_type='True',
                    parent=node,
                    depth=node.depth+1,
                    conjuntion_str=node.conjuntion_str + str(best_literal) + 'True',
                    significance_level=node.significance_level,
                    across_objects_bias=node.across_objects_bias,
                    consisten_dim_bias=node.consisten_dim_bias,
                    best_literal_criteria=node.best_literal_criteria,
                    valid_actions=node.valid_actions
                    )
                false_branch = Node(
                    root_action=node.root_action,
                    refinements=children_refinements,
                    max_depth=node.max_depth,
                    min_sample_size=node.min_sample_size,
                    node_type='False',
                    parent=node,
                    depth=node.depth+1,
                    conjuntion_str=node.conjuntion_str + str(best_literal) + 'False',
                    significance_level=node.significance_level,
                    across_objects_bias=node.across_objects_bias,
                    consisten_dim_bias=node.consisten_dim_bias,
                    best_literal_criteria=node.best_literal_criteria,
                    valid_actions=node.valid_actions
                    )
                # Inherit q-values from parent node.
                if inherit_q_values:
                    true_branch.q_value = node.q_value
                    false_branch.q_value = node.q_value
                # Make reference to children in parent node.
                node.true_branch = true_branch
                node.false_branch = false_branch
                node.children = [node.true_branch, node.false_branch]
            else:
                raise ValueError('Unrecognized literal type!')
            node.literal = best_literal
        return f_test
    
    @staticmethod
    def get_str_literal(literal):
        # literal_tuple = (literal.name, literal.obj1, literal.obj2)
        # return '(' + ', '.join(literal_tuple) + ')'
        return f"({literal.name}, {literal.obj1}, {literal.obj2})"

    def add_tree_nodes(self, action_str, dot):
        """Add recursively nodes of a tree to a graphviz Digraph (dot)."""
        current_node = self

        # Add edge from action to root.
        if current_node.node_type == 'root':
            dot.node(str(action_str), str(action_str))
            root_literal = str(round(current_node.q_value, 4)) if current_node.literal is None else self.get_str_literal(current_node.literal)
            dot.node(current_node.conjuntion_str, root_literal)
            dot.edge(action_str, current_node.conjuntion_str)
        
        # Base case: we've reached a leaf.
        root_literal = str(round(current_node.q_value, 8)) if current_node.literal is None else self.get_str_literal(current_node.literal)
        dot.node(current_node.conjuntion_str, root_literal)
        if current_node.parent is not None:
            dot.edge(current_node.parent.conjuntion_str, current_node.conjuntion_str, label=current_node.node_type)
        
        # Recursive call.
        for node in current_node.children:
            current_node = node
            current_node.add_tree_nodes(action_str, dot)                
        
        return

class RRLAgent():
    """A RRLAgent based on more/same/less partitions."""
    def __init__(
        self,
        env_name,
        run,
        save_dir,
        alpha=None,
        gamma=None,
        epsilon=None,
        epsilon_min=None,
        epsilon_decay=None,
        max_depth=None,
        inherit_q_values=None,
        significance_level=None,
        min_sample_size=None,
        action_buffer_capacity=None,
        across_objects_bias=None,
        consisten_dim_bias=None,
        best_literal_criteria=None,
        splits=None,
        include_non_informative_states=None,
        reward_manipulation=None,
        ):
        # Environment parameters.
        self.env_name = env_name
        self.env = gym.make(env_name)

        # Save info.
        self.run = run
        self.save_dir = save_dir

        # Training parameters.
        self.alpha = alpha if alpha else 0.1 # learning rate
        self.gamma = gamma if gamma else 0.99 # discount factor
        self.epsilon = epsilon if epsilon else 1.0 # exploration probability at start
        self.epsilon_min = epsilon_min if epsilon_min else 0.1 # minimum exploration probability
        self.epsilon_decay = epsilon_decay if epsilon_decay else 0.0000075  # exponential decay rate for exploration prob
        self.max_depth = max_depth if max_depth else 6
        self.inherit_q_values = inherit_q_values if inherit_q_values else True

        # Lists to keep track of useful statistics.
        self.training_session_data = []
        self.test_session_data = []

        # Define valid actions
        if self.env_name.startswith('Breakout'):
            self.valid_actions = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
        elif self.env_name.startswith('Pong'):
            self.valid_actions = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
        elif self.env_name.startswith('DemonAttack'):
            self.valid_actions = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
        else:
            raise ValueError('Unrecognized game!')
        
        # Define action space depending on environment.
        # self.action_space = self.env.unwrapped.get_action_meanings()
        self.action_space = self.valid_actions
        self.action_size = len(self.action_space)
        self.action_to_index = {value:count for count, value in enumerate(self.action_space)}
        self.index_to_action = {v:k for k,v in self.action_to_index.items()}

        self.splits = splits if splits else 'comparative'

        # Define possible refinements depending on game.
        if self.env_name.startswith('Breakout'):
            self.refinements = self.get_all_relations_breakout()
        elif self.env_name.startswith('Pong'):
            self.refinements = self.get_all_relations_pong()
        elif self.env_name.startswith('DemonAttack'):
            self.refinements = self.get_all_relations_demon_atack()
        else:
            raise ValueError('Unrecognized game!')

        # Set biases.
        self.across_objects_bias = across_objects_bias if across_objects_bias else False
        self.consisten_dim_bias = consisten_dim_bias if consisten_dim_bias else False

        # Best literal criteria
        self.best_literal_criteria = best_literal_criteria if best_literal_criteria else 'f-ratio'

        # Make root nodes based on possible actions and refinements.
        self.significance_level = significance_level if significance_level else 0.01
        self.min_sample_size = min_sample_size if min_sample_size else 10000
        self.root_nodes = self.make_root_nodes()
        # for k, v in self.root_nodes.items():
        #     print(k)
        # input()
        self.best_trees = {}

        # Define preprocessor
        if self.env_name.startswith('Breakout'):
            self.preprocessor = BreakoutPreprocessor()
        elif self.env_name.startswith('Pong'):
            self.preprocessor = PongPreprocessor()
        elif self.env_name.startswith('DemonAttack'):
            self.preprocessor = DemonAttackPreprocessor()
        else:
            raise ValueError('Unrecognized game!')
        
        # Define relational state function:
        if self.env_name.startswith('Breakout'):
            if self.splits == 'comparative':
                self.get_state = get_state_comparative_breakout
            else:
                self.get_state = get_state_logical_breakout
        elif self.env_name.startswith('Pong'):
            if self.splits == 'comparative':
                self.get_state = get_state_comparative_pong
            else:
                self.get_state = get_state_logical_pong
        elif self.env_name.startswith('DemonAttack'):
            if self.splits == 'comparative':
                self.get_state = get_state_comparative_demon_attack
            elif self.splits == 'logical':
                self.get_state = get_state_logical_demon_attack
        else:
            raise ValueError('Unrecognized game!')
        self.include_non_informative_states = include_non_informative_states if include_non_informative_states else False

        # Instantiate action buffer.
        self.action_buffer_capacity = action_buffer_capacity if action_buffer_capacity else 10
        self.action_buffer = Buffer(capacity=self.action_buffer_capacity)
        self.reward_buffer = Buffer(capacity=300)
        # Reward manipulation for Pong
        self.reward_manipulation = reward_manipulation if reward_manipulation else None

        # Close environment
        self.env.close()        

    def make_root_nodes(self):
        root_nodes = {}
        for action in self.valid_actions:
            node = Node(
                root_action=action,
                refinements=self.refinements,
                max_depth=self.max_depth,
                min_sample_size=self.min_sample_size,
                significance_level=self.significance_level,
                across_objects_bias=self.across_objects_bias,
                consisten_dim_bias=self.consisten_dim_bias,
                best_literal_criteria=self.best_literal_criteria,
                valid_actions=self.valid_actions
                )
            root_nodes[action] = node
        return root_nodes

    def get_all_relations_breakout(self):
        # Literal = namedtuple('Literal', ['name', 'type', 'obj1', 'obj2', 'obj3'], defaults=[None, None])
        # Get combinations of objects and order according to object_hierarchy.
        object_hierarchy = ['player', 'ball']
        objects_t0 = [x + '_t' for x in object_hierarchy]
        objects_tminus1 = [x + '_t-1' for x in object_hierarchy]
        object_combinations = list(itertools.combinations(objects_t0, 2))
        object_time_pairs = [(x, y) for (x, y) in zip(objects_t0, objects_tminus1)]
        # Get relations.
        if self.splits == 'comparative':
            dimensions = ['x', 'y']
        elif self.splits == 'logical':
            dimensions = ['same-x', 'more-x', 'less-x', 'same-y', 'more-y', 'less-y']
        all_literals = []
        for dimension in dimensions:
            for pair in object_combinations:
                all_literals.append(Literal(name=dimension, type=self.splits, obj1=pair[0], obj2=pair[1]))
            for pair in object_time_pairs:
                all_literals.append(Literal(name=dimension, type=self.splits, obj1=pair[0], obj2=pair[1]))
        # Add contact relation
        all_literals.append(Literal(name='incontact', type='logical', obj1='player_t', obj2='ball_t'))
        # Add objects presence
        all_literals.append(Literal(name='present', type='logical', obj1='player_t'))
        all_literals.append(Literal(name='present', type='logical', obj1='ball_t'))
        # Delete player trajectory relation
        if self.splits == 'comparative':
            tx_p = Literal(name='x', type=self.splits, obj1='player_t', obj2='player_t-1', obj3=None)
            all_literals.remove(tx_p)
        elif self.splits == 'logical':
            for name in ['more-x', 'same-x', 'less-x']:
                tx_p = Literal(name=name, type=self.splits, obj1='player_t', obj2='player_t-1', obj3=None)
                all_literals.remove(tx_p)
        return all_literals

    def get_all_relations_pong(self):
        # Literal = namedtuple('Literal', ['name', 'type', 'obj1', 'obj2', 'obj3'], defaults=[None, None])
        # Get combinations of objects and order according to object_hierarchy.
        object_hierarchy = ['player', 'ball', 'enemy']
        objects_t0 = [x + '_t' for x in object_hierarchy]
        objects_tminus1 = [x + '_t-1' for x in object_hierarchy]
        object_combinations = list(itertools.combinations(objects_t0, 2))
        object_time_pairs = [(x, y) for (x, y) in zip(objects_t0, objects_tminus1)]
        # Get relations.
        if self.splits == 'comparative':
            dimensions = ['x', 'y']
        elif self.splits == 'logical':
            dimensions = ['same-x', 'more-x', 'less-x', 'same-y', 'more-y', 'less-y']
        all_literals = []
        for dimension in dimensions:
            for pair in object_combinations:
                literal = Literal(name=dimension, type=self.splits, obj1=pair[0], obj2=pair[1])
                all_literals.append(literal)
            for pair in object_time_pairs:
                literal = Literal(name=dimension, type=self.splits, obj1=pair[0], obj2=pair[1])
                all_literals.append(literal)
        # Add contact relation
        literal = Literal(name='incontact', type='logical', obj1='player_t', obj2='ball_t')
        all_literals.append(literal)
        literal = Literal(name='incontact', type='logical', obj1='player_t', obj2='enemy_t')
        all_literals.append(literal)
        literal = Literal(name='incontact', type='logical', obj1='ball_t', obj2='enemy_t')
        all_literals.append(literal)
        # Delete player trajectory relation
        if self.splits == 'comparative':
            ty_p = Literal(name='y', type=self.splits, obj1='player_t', obj2='player_t-1', obj3=None)
            all_literals.remove(ty_p)
            # player_enemy_y = Literal(name='y', type=self.splits, obj1='player_t', obj2='enemy_t')
            # all_literals.remove(player_enemy_y)
        elif self.splits == 'logical':
            for name in ['more-y', 'same-y', 'less-y']:
                ty_p = Literal(name=name, type=self.splits, obj1='player_t', obj2='player_t-1', obj3=None)
                all_literals.remove(ty_p)
                # player_enemy_y = Literal(name=name, type=self.splits, obj1='player_t', obj2='enemy_t', obj3=None)
                # all_literals.remove(player_enemy_y)
        return all_literals

    def get_all_relations_demon_atack(self):
        # Get combinations of objects and order according to object_hierarchy.
        object_hierarchy = [
        'player',
        'enemy_missile',
        'enemy_big_0',
        'enemy_big_1',
        'enemy_big_2',
        'enemy_small_0',
        'enemy_small_1',
        'enemy_small_2',
        'enemy_small_3',
        'enemy_small_4',
        'enemy_small_5'
        ]
        objects_t0 = [x + '_t' for x in object_hierarchy]
        objects_tminus1 = [x + '_t-1' for x in object_hierarchy]
        # object_combinations = list(itertools.combinations(objects_t0, 2))

        object_combinations = [
            # ('player_t', 'player_missile_t'),
            ('player_t', 'enemy_missile_t'),
            ('player_t', 'enemy_big_0_t'),
            ('player_t', 'enemy_big_1_t'),
            ('player_t', 'enemy_big_2_t'),
            ('player_t', 'enemy_small_1_t'),
            ('player_t', 'enemy_small_2_t'),
            ('player_t', 'enemy_small_3_t'),
            ('player_t', 'enemy_small_4_t'),
            ('player_t', 'enemy_small_5_t')
            ]
        # object_time_pairs = [(x, y) for (x, y) in zip(objects_t0, objects_tminus1)]
        # Get relations.
        if self.splits == 'comparative':
            dimensions = ['x', 'y']
        elif self.splits == 'logical':
            dimensions = ['same-x', 'more-x', 'less-x', 'same-y', 'more-y', 'less-y']
        all_literals = []
        for dimension in dimensions:
            for pair in object_combinations:
                literal = Literal(name=dimension, type=self.splits, obj1=pair[0], obj2=pair[1])
                all_literals.append(literal)
            # for pair in object_time_pairs:
            #     literal = Literal(name=dimension, type=self.splits, obj1=pair[0], obj2=pair[1])
            #     all_literals.append(literal)
        # # Delete player trajectory relation
        # if self.splits == 'comparative':
        #     tx_p = Literal(name='x', type=self.splits, obj1='player_t', obj2='player_t-1')
        #     all_literals.remove(tx_p)
        # elif self.splits == 'logical':
        #     for name in ['more-x', 'same-x', 'less-x']:
        #         tx_p = Literal(name=name, type=self.splits, obj1='player_t', obj2='player_t-1')
        #         all_literals.remove(tx_p)
        
        return all_literals

    def save_trees(self):
        """Saves best and last trees. Naming convention: 'filename.pkl'."""
        
        file_best = f"{self.save_dir}train/{self.env_name}_run_{self.run}_best_trees.pkl"
        with open(file_best, 'wb') as handle:
            pickle.dump(self.best_trees, handle, pickle.HIGHEST_PROTOCOL)

        file_last = f"{self.save_dir}train/{self.env_name}_run_{self.run}_last_trees.pkl"
        with open(file_last, 'wb') as handle:
            pickle.dump(self.root_nodes, handle, pickle.HIGHEST_PROTOCOL)
    
    def save_current_tree(self, iteration):
        file_last = f"{self.save_dir}train/run_{self.run}/{self.env_name}_run_{self.run}_iter_{iteration}_trees.pkl"
        with open(file_last, 'wb') as handle:
            pickle.dump(self.root_nodes, handle, pickle.HIGHEST_PROTOCOL)

    def load_trees(self, filename):
        """Naming convention: 'filename.pkl'."""
        with open(filename, 'rb') as handle:
            self.root_nodes = pickle.load(handle)
            self.best_trees = self.root_nodes
    
    def get_dot_representation(self, best=False):
        """
        Get dot representation of the agent's current tree (default).
        If best == true return the representation of the best tree.
        """
        # Initialize graph.
        dot = graphviz.Digraph(node_attr={'shape': 'plaintext'})
        dot.graph_attr['rankdir'] = "TB"
        
        # Iterate over roots.
        if not best:
            for action_str, root_node in self.root_nodes.items():
                root_node.add_tree_nodes(action_str, dot)
        else:
            for action_str, root_node in self.best_trees.items():
                root_node.add_tree_nodes(action_str, dot)

        return dot
    
    def get_best_action(self, state):
        """Returns action symbol of the maximizing action given the state."""
        # If the state is empty return random action.
        if not state: #Fact(name='present', type='logical', value='False', obj1='ball_t') in state:
            # return self.index_to_action[self.env.action_space.sample()]
            return random.choice(self.valid_actions)
        else:
            actions_and_q_values = []
            for action, node in self.root_nodes.items():
                q_value = node.predict(state)
                actions_and_q_values.append((action, q_value))

            return max(actions_and_q_values, key=lambda item: item[1])[0]
    
    def epsilon_greedy_policy(self, state):
        """Epsilon-greedy policy based on the current q-value function and epsilon."""
        # If the state is empty return random action.
        if not state:
            # return self.env.action_space.sample()
            return self.action_to_index[random.choice(self.valid_actions)]
        else:
            max_action = self.get_best_action(state)
            max_action_index = self.action_to_index[max_action]

            action_probabilities = np.ones(self.action_size, dtype=float) * self.epsilon / self.action_size
            action_probabilities[max_action_index] += (1.0 - self.epsilon)
            action_index = np.random.choice(self.action_size, p=action_probabilities)
            return action_index

    def resample_repeated_action(self, action_index):
        # Append action to buffer.
        self.action_buffer.append(action_index)
        # Resample if the action has been repeated n times.
        # return self.env.action_space.sample() if self.action_buffer.test_all_same() else action_index
        return self.action_to_index[random.choice(self.valid_actions)] if self.action_buffer.test_all_same() else action_index

    def relational_q_learning(self, n_iterations, render=False, save_every=200000):
        # Initialize variables
        self.env = gym.make(self.env_name)
        i_episode = 0
        i_step = 0
        n_splits = 0
        cum_reward = 0        
        # Create filename
        train_file = f"{self.save_dir}train/{self.env_name}_run_{self.run}_train.csv"
        header = [
            'Game', 
            'Include non informative states', 
            'Run', 
            'Episode', 
            'Iteration', 
            'Reward', 
            'Cumulative reward', 
            'Split flag',
            'Number of splits', 
            'Tree'
            ]
        # Open the file in write mode
        with open(train_file, 'w') as f:
            # Create the csv writer
            writer = csv.writer(f)
            # Write header to the csv file
            writer.writerow(header)
            # Main loop (episodes)
            while i_step < n_iterations:            
                # Variables
                episode_return = 0
                episode_return2 = 0
                # Reset buffer
                self.action_buffer.clear()
                self.reward_buffer.clear()
                # Reset environment
                observation = self.env.reset()
                # Get relational state
                previous_info = self.preprocessor.get_info(observation)
                relational_state = self.get_state(
                    previous_info, 
                    None, 
                    self.include_non_informative_states
                    )
                # Episode loop
                for t in itertools.count():
                    if render:
                        self.env.render()
                    
                    # Act and get next relations
                    action_index = self.epsilon_greedy_policy(relational_state)
                    action_index = self.resample_repeated_action(action_index)
                    # action_symbol = self.index_to_action[action_index]
                    
                    root_node = self.root_nodes[self.index_to_action[action_index]]
                    observation, reward, done, info = self.env.step(action_index)

                    if self.env_name.startswith('DemonAttack'):
                        self.reward_buffer.append(reward)
                        if not done:
                            done = self.reward_buffer.test_all_zero()
                    episode_return2 += reward
                    info = self.preprocessor.get_info(observation)
                    if self.reward_manipulation == 'sign':
                        reward = np.sign(reward)
                    elif self.reward_manipulation == 'positive':
                        reward = 1 if reward > 0 else 0
                    elif self.reward_manipulation == 'duration':
                        reward = np.sign(reward) + 0.1
                    elif self.reward_manipulation == 'demon':
                        if self.env_name.startswith('DemonAttack'):
                            x_rel = Fact(name='x', type='comparative', value='same', obj1='player_t', obj2='enemy_missile_t')
                            y_rel = Fact(name='y', type='comparative', value='same', obj1='player_t', obj2='enemy_missile_t')
                            in_contact = -10 if (x_rel in relational_state and y_rel in relational_state) else 0
                            reward = reward + in_contact
                        else:
                            raise ValueError('This reward only makes sense for Demon Attack!')
                    else:
                        pass
                    next_relational_state = self.get_state(
                        info, 
                        previous_info, 
                        self.include_non_informative_states
                        )
                    # TD update
                    next_root_node = self.root_nodes[self.get_best_action(relational_state)]
                    q_value = next_root_node.predict(relational_state)
                    td_target = reward + self.gamma * q_value
                    td_delta = td_target - q_value
                    state_node = root_node.get_state_leaf(relational_state)
                    state_node.q_value += self.alpha * td_delta
                    # Update tree stats
                    root_node.update_statistics(relational_state)
                    # Run node split iteration
                    split_flag = root_node.split_iteration(
                        relational_state, 
                        inherit_q_values=self.inherit_q_values
                        )
                    # Get dot representation of tree
                    if split_flag:
                        dot = self.get_dot_representation()
                        dot_source = dot.source
                        n_splits += 1
                    else:
                        dot_source = ''
                    # Reassign current to next
                    relational_state = next_relational_state
                    previous_info = info
                    i_step += 1
                    if i_step % save_every == 0:
                        self.save_current_tree(i_step)
                        dot = self.get_dot_representation()
                        dot.render(
                            f"{self.save_dir}train/run_{self.run}/{self.env_name}_run_{self.run}_iter_{i_step}_dot.gv", 
                            view=False
                            )
                    if i_step >= n_iterations:
                        break
                    # Update explorationm rate
                    if self.epsilon > self.epsilon_min:
                        self.epsilon *= (1-self.epsilon_decay)
                        if self.epsilon < self.epsilon_min:
                            self.epsilon = self.epsilon_min
                    # Save info to file
                    cum_reward += reward
                    writer.writerow([
                        self.env_name, 
                        self.include_non_informative_states, 
                        self.run, 
                        i_episode, 
                        i_step, 
                        reward, 
                        cum_reward, 
                        split_flag,
                        n_splits, 
                        dot_source
                        ])
                    # Update return
                    episode_return += reward
                    # At the end of each episode
                    if done:
                        i_episode += 1
                        break
                # Print info every n episodes
                if i_episode % 10 == 0:
                    print(
                        'exploration_rate:', round(self.epsilon, 3),
                        'iteration:', i_step,
                        'episode', i_episode,
                        'ep_return', episode_return2,
                        'ep_timesteps', t,
                        'n_splits:', n_splits)
        # Close the environment
        self.env.close()
        return
    
    def test_trees(self, n_episodes=10, epsilon_test=0.05, render=False, save_imgs_dir=None):
        # Make new environment
        self.env = gym.make(self.env_name)
        # Set agent exploration rate
        self.epsilon = epsilon_test
        # Initialize variables
        i = 0
        all_returns = []
        # Main loop
        for i_episode in range(n_episodes):
            # Reset environment
            observation = self.env.reset()
            # Get relational state
            previous_info = self.preprocessor.get_info(observation)
            relational_state = self.get_state(previous_info, None, self.include_non_informative_states)
            # Episode loop
            episode_return = 0
            for t in itertools.count():
                if render:
                    self.env.render()
                # Act and get next relations
                action_index = self.epsilon_greedy_policy(relational_state)
                observation, reward, done, info = self.env.step(action_index)
                if save_imgs_dir:
                    img = Image.fromarray(observation, 'RGB')
                    img.save(save_imgs_dir + f'/{i}.png')
                    i += 1
                info = self.preprocessor.get_info(observation)
                next_relational_state = self.get_state(info, previous_info, self.include_non_informative_states)
                # Reassign current to next.
                relational_state = next_relational_state
                previous_info = info
                # Update return
                episode_return += reward
                # At the end of each episode
                if done:
                    # Append returns
                    all_returns.append(episode_return)
                    break
        # Close the environment
        self.env.close()
        return np.mean(all_returns)
    
    def get_best_trees_path(self, n_episodes=10, epsilon_test=0.05, render=False):
        trees_dir = f"{self.save_dir}train/run_{self.run}"
        best_return = -math.inf
        best_trees_path = None
        for root, dirs, files in os.walk(trees_dir):
            for name in files:
                filepath = os.path.join(root, name)
                if filepath.endswith('trees.pkl'):
                    self.load_trees(filepath)
                    trees_return = self.test_trees(
                        n_episodes=n_episodes,
                        epsilon_test=epsilon_test,
                        render=render)
                    if trees_return >= best_return:
                        best_trees_path = filepath
                        best_return = trees_return
                        # Save pdf of best trees
                        best_dot = self.get_dot_representation()
                        best_dot.render(
                            f"{self.save_dir}train/{self.env_name}_run_{self.run}_best_dot.gv", 
                            view=False
                            )
        return best_trees_path

    def test_agent(self, n_episodes=100, epsilon_test=0.05, render=False):
        # Get best trees and load
        best_trees_path = self.get_best_trees_path(
            n_episodes=10, 
            epsilon_test=0.05, 
            render=False
            )
        self.load_trees(best_trees_path)
        # Make new environment
        self.env = gym.make(self.env_name)
        # Get the best tree during training.
        self.root_nodes = self.best_trees
        # Set agent exploration rate.
        self.epsilon = epsilon_test
        # Initialize variables.
        all_returns = []
        # Main loop.
        for i_episode in range(n_episodes):
            # Reset environment
            observation = self.env.reset()
            # Get relational state
            previous_info = self.preprocessor.get_info(observation)
            relational_state = self.get_state(
                previous_info, 
                None, 
                self.include_non_informative_states
                )
            # Episode loop
            episode_return = 0
            for t in itertools.count():
                if render:
                    self.env.render()
                # Act and get next relations.
                action_index = self.epsilon_greedy_policy(relational_state)
                observation, reward, done, info = self.env.step(action_index)
                info = self.preprocessor.get_info(observation)
                next_relational_state = self.get_state(
                    info, 
                    previous_info, 
                    self.include_non_informative_states
                    )
                # Reassign current to next.
                relational_state = next_relational_state
                previous_info = info
                # Update return.
                episode_return += reward
                # At the end of each episode.
                if done:
                    # Append returns
                    all_returns.append(episode_return)
                    break
            
            print(round(self.epsilon, 3), i_episode, episode_return)
        # Get stats
        all_returns = np.array(all_returns)
        # Create files
        test_file = f"{self.save_dir}test/{self.env_name}_run_{self.run}_test.csv"
        test_data = {
            'Game': [self.env_name], 
            'Include incomplete states': [self.include_non_informative_states],
            'Reward manipulation': [self.reward_manipulation],
            'Run': [self.run],
            'Mean return': [np.mean(all_returns)],
            'Std Return': [np.std(all_returns)],
            'Min return': [np.amin(all_returns)],
            'Max return': [np.amax(all_returns)]
            }
        df = pd.DataFrame(test_data)
        df.to_csv(test_file)
        test_file_raw = f"{self.save_dir}test/{self.env_name}_run_{self.run}_test_raw.csv"
        test_data_raw = {
            'Game': [self.env_name] * len(all_returns), 
            'Include incomplete states': [self.include_non_informative_states] * len(all_returns), 
            'Reward manipulation': [self.reward_manipulation] * len(all_returns),
            'Run': [self.run] * len(all_returns),
            'Returns': all_returns
            }
        df2 = pd.DataFrame(test_data_raw)
        df2.to_csv(test_file_raw)
        # Close the environment
        self.env.close()
        return

