import os
import math
import itertools
import pickle
import csv
import graphviz
import numpy as np
import pandas as pd
from PIL import Image
from collections import namedtuple
from scipy.stats import f
import gymnasium as gym
from gymnasium import Wrapper
from gymnasium.wrappers import AtariPreprocessing, TransformReward
import ale_py
from utilities import Buffer
import preprocessing


# Literal class. The last two arguments are optional
Literal = namedtuple('Literal', ['name', 'type', 'obj1', 'obj2', 'obj3'], defaults=[None, None])

# Gymnasium environment wrapper
class FireReset(Wrapper):
    """
    Press FIRE on reset.
    """
    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE', (
            'Only use fire reset wrapper for suitable environment!')
        assert len(env.unwrapped.get_action_meanings()) >= 3, (
            'Only use fire reset wrapper for suitable environment!')
    
    def reset(self, *, seed):
        """gym.Env reset function.

        Args:
            kwargs (dict): extra arguments passed to gym.Env.reset()

        Returns:
            np.ndarray: next observation.
        """
        self.env.reset(seed=seed)
        observation, _, terminated, truncated, info = self.env.step(1)
        done = terminated or truncated
        if done:
            self.env.reset(seed=seed)
        observation, _, terminated, truncated, info = self.env.step(2)
        done = terminated or truncated
        if done:
            self.env.reset(seed=seed)
        return observation, info

class Node():
    def __init__(
        self,
        root_action,
        refinements=None,
        max_depth=10,
        min_sample_size=100000,
        depth=None,
        node_type=None,
        parent=None,
        literal=None,
        conjuntion_str=None,
        significance_level=0.001,
        best_literal_criteria='p-value',
        valid_actions=None
        ):
        # Always require a root action
        self.root_action = root_action
        # Define node possible refinements
        self.refinements = refinements if refinements else []
        # Free parameters
        self.max_depth = max_depth
        self.min_sample_size = min_sample_size
        self.significance_level = significance_level # significance level for f-test
        # Default current depth of node
        self.depth = depth if depth else 0
        # Type of node
        self.node_type = node_type if node_type else 'root'
        # Node parent
        self.parent = parent if parent else None
        # Literal that splits the node
        self.literal = literal if literal else None
        # Initialize q-value
        self.q_value = 0.0
        # Initialize statistics dictionary
        self.refinements_stats = self.make_stats_dict()
        # Always initialize the more, same and less braches as None
        self.more_branch = None 
        self.same_branch = None
        self.less_branch = None
        self.children = []
        # Get node string identifier for graphviz
        self.conjuntion_str = conjuntion_str if conjuntion_str else self.root_action + 'root'
        # Best literal criteria
        self.best_literal_criteria = best_literal_criteria
        # Action to consider when spliting
        self.valid_actions = valid_actions if valid_actions else []

    def make_stats_dict(self):
        """Makes a statistics dictionary to base the splits on."""
        stats = {}
        for literal in self.refinements:
            # Comparative literal example: ('x', 'paddle', 'ball')
            if literal.type == 'comparative':
                literals_dict = {
                    literal: {
                        'Total': {'n': 0, 'mu': 0, 'J': 0},
                        'same': {'n': 0, 'mu': 0, 'J': 0},
                        'more': {'n': 0, 'mu': 0, 'J': 0},
                        'less': {'n': 0, 'mu': 0, 'J': 0}
                        }}
            # Logical literal example: ('more-x', 'paddle', 'ball')
            elif literal.type == 'logical':
                literals_dict = {
                    literal: {
                        'Total': {'n': 0, 'mu': 0, 'J': 0},
                        'True': {'n': 0, 'mu': 0, 'J': 0},
                        'False': {'n': 0, 'mu': 0, 'J': 0}
                    }}
            else:
                raise ValueError("Unrecognized literal type!")
            stats.update(literals_dict)
        return stats

    def get_f_ratio(self, literal):
        """Returns the ratio of the variance of the q-values of the examples
        collected in a leaf before and after splitting."""
        # Because J = sigma * n, dividing J by n_{Total} gives the weighted variance
        try:
            weighted_sigma_same = self.refinements_stats[literal]['same']['J'] / self.refinements_stats[literal]['Total']['n']
            weighted_sigma_more = self.refinements_stats[literal]['more']['J'] / self.refinements_stats[literal]['Total']['n']
            weighted_sigma_less = self.refinements_stats[literal]['less']['J'] / self.refinements_stats[literal]['Total']['n']
            sigma_overall = self.refinements_stats[literal]['Total']['J'] / self.refinements_stats[literal]['Total']['n']
            return (weighted_sigma_same + weighted_sigma_more + weighted_sigma_less) / sigma_overall
        except KeyError:
            weighted_sigma_true = self.refinements_stats[literal]['True']['J'] / self.refinements_stats[literal]['Total']['n']
            weighted_sigma_false = self.refinements_stats[literal]['False']['J'] / self.refinements_stats[literal]['Total']['n']
            sigma_overall = self.refinements_stats[literal]['Total']['J'] / self.refinements_stats[literal]['Total']['n']
            return (weighted_sigma_true + weighted_sigma_false) / sigma_overall

    def get_max_q_value(self, literal):
        """Get the maximum q-value of the node."""
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
        
        # Select literals that have reached the minimum number of samples
        valid_literals = [x for x in self.refinements if self.refinements_stats[x]['Total']['n'] > self.min_sample_size]

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
        df2 = m - 1
        # One-tailed "less" F-test
        p_value = f.cdf(f_ratio, df1, df2)
        return p_value <= self.significance_level

    def predict(self, state):
        """
        Predicts a q-value for a state by traversing the tree until
        the last branch.
        This method should be called from a root node.
        """
        # Initialize current node
        current_node = self

        # Base case: we've reached a leaf
        if current_node.literal is None:
            return current_node.q_value

        # Base case 2: state does not have any facts
        if not state:
            return current_node.q_value
        
        # Traversing the nodes all the way to the bottom
        while current_node.literal is not None:
            # Get relevant literal
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

    def update_statistics(self, state):
        """
        Update the statistics of the leaf node corresponding to the state.
        This method should be called from a root node only.
        Update equations:
            n_t = n_{t-1} + 1
            mu_t = mu_{t-1} + (q_t - mu_{t-1})/n_t
            J_t = J_{t-1} + (q_t - mu_{t-1}) * (q_t - mu_t)
        """
        
        # If there are no relations in the state return inmediatly
        if not state:
            return
        # Traverse the nodes all the way to the bottom
        current_node = self.get_state_leaf(state)
        for literal in current_node.refinements:
            relevant_facts = [x for x in state if x.name == literal.name and x.obj1 == literal.obj1 and x.obj2 == literal.obj2]
            if relevant_facts:
                # Always update total
                # Update n
                current_node.refinements_stats[literal]['Total']['n'] += 1
                # Update mean
                n = current_node.refinements_stats[literal]['Total']['n']
                old_mean = current_node.refinements_stats[literal]['Total']['mu']
                current_node.refinements_stats[literal]['Total']['mu'] = old_mean + (current_node.q_value - old_mean) / n
                # Update J (variance times n)
                new_mean = current_node.refinements_stats[literal]['Total']['mu']
                old_s = current_node.refinements_stats[literal]['Total']['J']
                current_node.refinements_stats[literal]['Total']['J'] = \
                    old_s + (current_node.q_value - old_mean) * (current_node.q_value - new_mean)
                
                for fact in relevant_facts:
                    if fact.value == 'same':
                        # Update n
                        current_node.refinements_stats[literal]['same']['n'] += 1
                        # Update mean
                        n = current_node.refinements_stats[literal]['same']['n']
                        old_mean = current_node.refinements_stats[literal]['same']['mu']
                        current_node.refinements_stats[literal]['same']['mu'] = old_mean + (current_node.q_value - old_mean) / n
                        # Update J (variance times n)
                        new_mean = current_node.refinements_stats[literal]['same']['mu']
                        old_s = current_node.refinements_stats[literal]['same']['J']
                        current_node.refinements_stats[literal]['same']['J'] = \
                            old_s + (current_node.q_value - old_mean) * (current_node.q_value - new_mean)
                    elif fact.value == 'more':
                        # Update n
                        current_node.refinements_stats[literal]['more']['n'] += 1
                        # Update mean
                        n = current_node.refinements_stats[literal]['more']['n']
                        old_mean = current_node.refinements_stats[literal]['more']['mu']
                        current_node.refinements_stats[literal]['more']['mu'] = old_mean + (current_node.q_value - old_mean) / n
                        # Update J (variance times n)
                        new_mean = current_node.refinements_stats[literal]['more']['mu']
                        old_s = current_node.refinements_stats[literal]['more']['J']
                        current_node.refinements_stats[literal]['more']['J'] = \
                            old_s + (current_node.q_value - old_mean) * (current_node.q_value - new_mean)
                    elif fact.value == 'less':
                        # Update n
                        current_node.refinements_stats[literal]['less']['n'] += 1
                        # Update mean
                        n = current_node.refinements_stats[literal]['less']['n']
                        old_mean = current_node.refinements_stats[literal]['less']['mu']
                        current_node.refinements_stats[literal]['less']['mu'] = old_mean + (current_node.q_value - old_mean) / n
                        # Update J (variance times n)
                        new_mean = current_node.refinements_stats[literal]['less']['mu']
                        old_s = current_node.refinements_stats[literal]['less']['J']
                        current_node.refinements_stats[literal]['less']['J'] = \
                            old_s + (current_node.q_value - old_mean) * (current_node.q_value - new_mean)
                    elif fact.value == 'True':
                        # Update n
                        current_node.refinements_stats[literal]['True']['n'] += 1
                        # Update mean
                        n = current_node.refinements_stats[literal]['True']['n']
                        old_mean = current_node.refinements_stats[literal]['True']['mu']
                        current_node.refinements_stats[literal]['True']['mu'] = old_mean + (current_node.q_value - old_mean) / n
                        # Update J (variance times n)
                        new_mean = current_node.refinements_stats[literal]['True']['mu']
                        old_s = current_node.refinements_stats[literal]['True']['J']
                        current_node.refinements_stats[literal]['True']['J'] = \
                            old_s + (current_node.q_value - old_mean) * (current_node.q_value - new_mean)
                    elif fact.value == 'False':
                        # Update n
                        current_node.refinements_stats[literal]['False']['n'] += 1
                        # Update mean
                        n = current_node.refinements_stats[literal]['False']['n']
                        old_mean = current_node.refinements_stats[literal]['False']['mu']
                        current_node.refinements_stats[literal]['False']['mu'] = old_mean + (current_node.q_value - old_mean) / n
                        # Update S (variance times n)
                        new_mean = current_node.refinements_stats[literal]['False']['mu']
                        old_s = current_node.refinements_stats[literal]['False']['J']
                        current_node.refinements_stats[literal]['False']['J'] = \
                            old_s + (current_node.q_value - old_mean) * (current_node.q_value - new_mean)
                    else:
                        raise ValueError('Unrecognised comparative value!')
             
        return

    def split_iteration(self, state, inherit_q_values):
        """Splits the leaf node corresponding to the state if the
        best literal passes the F-test."""

        # If the root action is not part of the valid actions don't split
        if self.root_action not in self.valid_actions:
            return False
        
        # If there are no between-object relations in the state return inmediatly
        facts = [x for x in state if x.obj2 is not None]
        if not facts:
            return False
        
        # Find relevant node.
        node = self.get_state_leaf(state)

        # If node depth is equal or bigger than max_depth return inmediatly
        if node.depth >= self.max_depth:
            return False

        # If the node is not a leaf return inmediatly
        if node.literal is not None:
            return False
        
        # Find the best literal and F-value
        best_literal = node.find_best_literal()

        # If none of the relations has reached the minimun number of samples return inmediatly
        if best_literal is None:
            return False

        # Run F-test
        # n is the overall sample size
        n = node.refinements_stats[best_literal]['Total']['n']
        best_f_ratio = node.get_f_ratio(best_literal)
        f_test = self.f_test(best_f_ratio, n=n, m=n)

        # Split the node if f-test is postive
        if f_test:
            children_refinements = [x for x in node.refinements if x != best_literal]
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
                    best_literal_criteria=node.best_literal_criteria,
                    valid_actions=node.valid_actions
                    )
                # Inherit q-values from parent node
                if inherit_q_values:
                    more_branch.q_value = node.q_value 
                    same_branch.q_value = node.q_value 
                    less_branch.q_value = node.q_value 

                # Make reference to children in parent node
                node.more_branch = more_branch
                node.same_branch = same_branch
                node.less_branch = less_branch
                node.children = [node.more_branch, node.same_branch, node.less_branch]

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
                    best_literal_criteria=node.best_literal_criteria,
                    valid_actions=node.valid_actions
                    )
                # Inherit q-values from parent node
                if inherit_q_values:
                    true_branch.q_value = node.q_value
                    false_branch.q_value = node.q_value
                # Make reference to children in parent node
                node.true_branch = true_branch
                node.false_branch = false_branch
                node.children = [node.true_branch, node.false_branch]
            else:
                raise ValueError('Unrecognized literal type!')
            node.literal = best_literal
        return f_test
    
    @staticmethod
    def get_str_literal(literal):
        """Get string representation of literal."""
        return f"({literal.name}, {literal.obj1}, {literal.obj2})"

    def add_tree_nodes(self, action_str, dot):
        """Add recursively nodes of a tree to a graphviz Digraph (dot)."""
        current_node = self

        # Add edge from action to root
        if current_node.node_type == 'root':
            dot.node(str(action_str), str(action_str))
            root_literal = str(round(current_node.q_value, 4)) if current_node.literal is None else self.get_str_literal(current_node.literal)
            dot.node(current_node.conjuntion_str, root_literal)
            dot.edge(action_str, current_node.conjuntion_str)
        
        # Base case: we've reached a leaf
        root_literal = str(round(current_node.q_value, 8)) if current_node.literal is None else self.get_str_literal(current_node.literal)
        dot.node(current_node.conjuntion_str, root_literal)
        if current_node.parent is not None:
            dot.edge(current_node.parent.conjuntion_str, current_node.conjuntion_str, label=current_node.node_type)
        
        # Recursive call
        for node in current_node.children:
            current_node = node
            current_node.add_tree_nodes(action_str, dot)                
        
        return dot

    def add_literals(self, literal_list):
        """Add recursively split literals of a list."""
        current_node = self
        # Base case: we've reached a leaf
        literal = None if current_node.literal is None else current_node.literal
        if literal is not None:
            literal_list.append(literal)

        # Recursive call
        for node in current_node.children:
            current_node = node
            current_node.add_literals(literal_list)

        return literal_list

class RRLAgent():
    def __init__(
        self,
        env_name,
        game_name,
        run,
        save_dir,
        initial_seed,
        alpha=0.1,
        gamma=0.99,
        epsilon_init=1.0,
        epsilon_min=0.1,
        epsilon_decay_steps=500000,
        max_depth=10,
        inherit_q_values=True,
        significance_level=0.001,
        min_sample_size=100000,
        action_buffer_capacity=10,
        best_literal_criteria='p-value',
        splits='comparative',
        include_incomplete_states=False
        ):
        # Environment parameters
        self.env_name = env_name
        self.game_name = game_name
        # Save info
        self.run = run
        self.save_dir = save_dir
        # Set seed for reproducibility
        self.initial_seed = initial_seed
        self.rng = np.random.default_rng(seed=self.initial_seed)
        # Training parameters
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount factor
        self.epsilon = epsilon_init # current exploration probability
        self.epsilon_init = epsilon_init # exploration probability at start
        self.epsilon_min = epsilon_min # minimum exploration probability
        self.epsilon_decay_steps = epsilon_decay_steps # steps from epsilon_init to epsilon_min
        self.epsilon_decay_rate = self.calculate_multiplicative_decay_rate() # epsilon decay rate
        self.max_depth = max_depth # maximum depth of trees
        self.inherit_q_values = inherit_q_values # whether new leafs shoulf inherit q-values from thir parent node
        # Define valid actions
        if self.game_name.startswith('Breakout'):
            self.valid_actions = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
        elif self.game_name.startswith('Pong'):
            self.valid_actions = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
        elif self.game_name.startswith('DemonAttack'):
            self.valid_actions = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
        else:
            raise ValueError('Unrecognized game!')
        # Define action size and action dictionaries depending on environment
        self.action_size = len(self.valid_actions)
        self.action_to_index = {value:count for count, value in enumerate(self.valid_actions)}
        self.index_to_action = {v:k for k,v in self.action_to_index.items()}
        # Model version
        self.splits = splits if splits else 'comparative'
        # Define possible refinements depending on game
        if self.game_name.startswith('Breakout'):
            self.refinements = self.get_all_relations_breakout()
        elif self.game_name.startswith('Pong'):
            self.refinements = self.get_all_relations_pong()
        elif self.game_name.startswith('DemonAttack'):
            self.refinements = self.get_all_relations_demon_atack()
        else:
            raise ValueError('Unrecognized game!')
        # Best literal criteria
        self.best_literal_criteria = best_literal_criteria
        # Make root nodes based on possible actions and refinements
        self.significance_level = significance_level
        self.min_sample_size = min_sample_size
        self.root_nodes = self.make_root_nodes()
        self.best_trees = {}
        # Define preprocessor
        if self.game_name.startswith('Breakout'):
            self.preprocessor = preprocessing.BreakoutPreprocessor()
        elif self.game_name.startswith('Pong'):
            self.preprocessor = preprocessing.PongPreprocessor()
        elif self.game_name.startswith('DemonAttack'):
            self.preprocessor =preprocessing.DemonAttackPreprocessor()
        else:
            raise ValueError('Unrecognized game!')
        # Define relational state function
        if self.game_name.startswith('Breakout'):
            if self.splits == 'comparative':
                self.get_state = preprocessing.get_state_comparative_breakout
            else:
                self.get_state = preprocessing.get_state_logical_breakout
        elif self.game_name.startswith('Pong'):
            if self.splits == 'comparative':
                self.get_state = preprocessing.get_state_comparative_pong
            else:
                self.get_state = preprocessing.get_state_logical_pong
        elif self.game_name.startswith('DemonAttack'):
            if self.splits == 'comparative':
                self.get_state = preprocessing.get_state_comparative_demon_attack
            elif self.splits == 'logical':
                self.get_state = preprocessing.get_state_logical_demon_attack
        else:
            raise ValueError('Unrecognized game!')
        self.include_incomplete_states = include_incomplete_states if include_incomplete_states else False
        # Instantiate action and reward buffers
        self.action_buffer_capacity = action_buffer_capacity
        self.action_buffer = Buffer(capacity=self.action_buffer_capacity)

    def make_root_nodes(self):
        """Creates the root (action) nodes."""
        root_nodes = {}
        for action in self.valid_actions:
            node = Node(
                root_action=action,
                refinements=self.refinements,
                max_depth=self.max_depth,
                min_sample_size=self.min_sample_size,
                significance_level=self.significance_level,
                best_literal_criteria=self.best_literal_criteria,
                valid_actions=self.valid_actions
                )
            root_nodes[action] = node
        return root_nodes

    def get_all_relations_breakout(self):
        """Defines all possible refinements for Breakout."""
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
        """Defines all possible refinements for Pong."""
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
        elif self.splits == 'logical':
            for name in ['more-y', 'same-y', 'less-y']:
                ty_p = Literal(name=name, type=self.splits, obj1='player_t', obj2='player_t-1', obj3=None)
                all_literals.remove(ty_p)
        return all_literals

    def get_all_relations_demon_atack(self):
        """Defines all possible refinements for Demon Attack."""
        # Get combinations of objects and order according to object_hierarchy.
        object_combinations = [
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
        # Get relations
        if self.splits == 'comparative':
            dimensions = ['x', 'y']
        elif self.splits == 'logical':
            dimensions = ['same-x', 'more-x', 'less-x', 'same-y', 'more-y', 'less-y']
        all_literals = []
        for dimension in dimensions:
            for pair in object_combinations:
                literal = Literal(name=dimension, type=self.splits, obj1=pair[0], obj2=pair[1])
                all_literals.append(literal)
                
        return all_literals

    def save_current_tree(self, iteration):
        """Saves the current set of trees to a file."""
        file_last = f"{self.save_dir}train/run_{self.run}/{self.game_name}_run_{self.run}_iter_{iteration}_trees.pkl"
        with open(file_last, 'wb') as handle:
            pickle.dump(self.root_nodes, handle, pickle.HIGHEST_PROTOCOL)

    def load_trees(self, filename):
        """Loads a set of saved trees."""
        with open(filename, 'rb') as handle:
            self.root_nodes = pickle.load(handle)
            self.best_trees = self.root_nodes
    
    def get_dot_representation(self):
        """
        Get dot representation of the agent's current tree.
        """
        # Initialize graph.
        dot = graphviz.Digraph(node_attr={'shape': 'plaintext'})
        dot.graph_attr['rankdir'] = "TB"
        # Iterate over root nodes
        for action_str, root_node in self.root_nodes.items():
                dot = root_node.add_tree_nodes(action_str, dot)
        return dot
    
    def sample_valid_action(self):
        """Samples a random action."""
        return self.action_to_index[self.rng.choice(self.valid_actions)]
    
    def calculate_multiplicative_decay_rate(self):
        """
        Calculates the multiplicative decay rate for epsilon-greedy exploration.

        Args:
            self.epsilon_init (float): The starting value of epsilon.
            self.epsilon_min (float): The target ending value of epsilon.
            self.epsilon_decay_steps (int): The number of steps or episodes over which the decay occurs.

        Returns:
            float: The calculated multiplicative decay rate.
        """
        if self.epsilon_decay_steps <= 0:
            raise ValueError("Number of steps must be greater than 0.")
        if self.epsilon_init <= 0 or self.epsilon_min < 0:
            raise ValueError("Epsilon values must be non-negative, and epsilon_init must be positive.")
        if self.epsilon_min > self.epsilon_init:
            raise ValueError("Final epsilon cannot be greater than initial epsilon for decay.")

        decay_rate = (self.epsilon_min / self.epsilon_init)**(1 / self.epsilon_decay_steps)
        return decay_rate

    def get_best_action(self, state):
        """Returns action symbol of the maximizing action given the state."""
        # If state is empty choose a random action
        if not state:
            max_action = self.rng.choice(self.valid_actions)
        else:
            actions_and_q_values = []
            for action, node in self.root_nodes.items():
                q_value = node.predict(state)
                actions_and_q_values.append((action, q_value))
            max_action = max(actions_and_q_values, key=lambda item: item[1])[0]
        return max_action
    
    def epsilon_greedy_policy(self, state):
        """Epsilon-greedy policy based on the current q-value function and epsilon."""
        max_action = self.get_best_action(state)
        max_action_index = self.action_to_index[max_action]
        action_probabilities = np.ones(self.action_size, dtype=float) * self.epsilon / self.action_size
        action_probabilities[max_action_index] += (1.0 - self.epsilon)
        action_index = self.rng.choice(self.action_size, p=action_probabilities)
        return action_index

    def greedy_policy(self, state):
        """Deterministic greedy policy."""
        max_action = self.get_best_action(state)
        return self.action_to_index[max_action]
    
    def resample_repeated_action(self, action_index):
        """Returns a random action if all previous actions in the buffer are the same."""
        # Append action to buffer.
        self.action_buffer.append(action_index)
        # Resample if the action has been repeated n times
        new_action_index = self.sample_valid_action() if self.action_buffer.test_all_same() else action_index
        return new_action_index

    def terminate_episode_if_no_rewards(self, reward, done):
        """Terminates the episode if the last 300 rewards are 0."""
        self.reward_buffer.append(reward)
        if not done:
            done = self.reward_buffer.test_all_zero()
        return done
    
    def make_train_environment(self):
        """Sets the environment to press FIRE on reset and 
        applies the sign function to the reward."""
        env = gym.make(
            self.env_name, 
            obs_type="rgb",
            frameskip=4,
            repeat_action_probability=0.0,
            full_action_space=False)
        if self.game_name == 'Breakout' or self.game_name == 'DemonAttack':
            env = TransformReward(env, lambda r: np.sign(r))
        elif self.game_name == 'Pong':
            env = TransformReward(env, lambda r: np.sign(r) + 0.1)
        else:
            raise ValueError('Unrecognized game!')
        return FireReset(env)
    
    def make_test_environment(self):
        """Sets the environment to press FIRE on reset and 
        applies the sign function to the reward."""
        env = gym.make(
            self.env_name,
            obs_type="rgb",
            frameskip=4,
            repeat_action_probability=0.0,
            full_action_space=False)
        return FireReset(env)
    
    def relational_q_learning(self, n_iterations, print_every_episodes=1000):
        """Trains the agent and saves the results."""
        # Make environment
        env = self.make_train_environment()
        # Initialize variables
        i_episode = 0
        i_step = 0
        n_splits = 0
        best_return = -math.inf        
        # Create filename
        train_file = f"{self.save_dir}train/{self.game_name}_run_{self.run}_train.csv"
        header = [
            'Game', 
            'Run', 
            'Episode', 
            'Iteration', 
            'Reward', 
            'Number of splits', 
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
                # Reset buffer
                self.action_buffer.clear()
                # Reset environment
                episode_seed = int(self.rng.integers(30000))
                observation, info = env.reset(seed=episode_seed)
                env.action_space.seed(episode_seed)
                # Get relational state
                previous_info = self.preprocessor.get_info(observation)
                relational_state = self.get_state(
                    previous_info, 
                    None, 
                    self.include_incomplete_states
                    )
                # Episode loop
                for t in itertools.count():
                    # Act and get next relations
                    action_index = self.epsilon_greedy_policy(relational_state)
                    if self.game_name == "Breakout":
                        action_index = self.resample_repeated_action(action_index)
                    root_node = self.root_nodes[self.index_to_action[action_index]]
                    observation, reward, terminated, truncated, info = env.step(action_index)
                    # Episode ends if either terminated OR truncated
                    done = terminated or truncated
                    # Get info
                    info = self.preprocessor.get_info(observation)
                    # Get next state
                    next_relational_state = self.get_state(
                        info, 
                        previous_info, 
                        self.include_incomplete_states
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
                    # Update number of splits
                    if split_flag:
                        n_splits += 1
                    # Reassign current to next
                    relational_state = next_relational_state
                    previous_info = info
                    # Update iteration counter
                    i_step += 1
                    # Stop if maximum iterations has been reached
                    if i_step >= n_iterations:
                        break
                    # Update explorationm rate
                    self.epsilon = max(self.epsilon * self.epsilon_decay_rate, self.epsilon_min)
                    # Save info to file
                    writer.writerow([
                        self.game_name, 
                        self.run, 
                        i_episode, 
                        i_step, 
                        reward, 
                        n_splits
                        ])
                    # Update return
                    episode_return += reward
                    # At the end of each episode
                    if done:
                        i_episode += 1
                        if episode_return > best_return:
                            self.save_current_tree(i_step)
                            dot = self.get_dot_representation()
                            dot_filename = (
                                f"{self.save_dir}train/run_{self.run}/"
                                f"{self.game_name}_run_{self.run}_iter_{i_step}_dot.gv"
                                )
                            dot.render(dot_filename, view=False)
                            best_return = episode_return
                        break
                # Print info every n episodes
                if i_episode % print_every_episodes == 0:
                    print(
                        'iteration', i_step,
                        'episode', i_episode,
                        'epsilon:', round(self.epsilon, 3),
                        'return', episode_return,
                        'n_splits:', n_splits)
        # Close the environment
        env.close()
        return
    
    @staticmethod
    def get_iteration_number(pkl_file_pah):
        """Get iteration number of tree from pkl file path."""
        return int(pkl_file_pah.split('_')[4])

    def get_best_trees_path(self):
        """Get the best (i.e. last) set of trees saved and saves its dot representation."""
        trees_dir = f"{self.save_dir}train/run_{self.run}"
        for root, dirs, files in os.walk(trees_dir):
            pkl_files = [x for x in files if x.endswith('trees.pkl')]
            best_file_name = sorted(pkl_files, key=self.get_iteration_number)[-1]
            best_trees_path = os.path.join(root, best_file_name)
            self.load_trees(best_trees_path)
            # Save pdf of best trees
            best_dot = self.get_dot_representation()
            best_dot.render(
                f"{self.save_dir}train/{self.game_name}_run_{self.run}_best_dot.gv", 
                view=False
                )
        return best_trees_path
    
    def get_iteration_best_model(self):
        """Get the iteration number of the best set of trees saved."""
        trees_dir = f"{self.save_dir}train/run_{self.run}"
        for root, dirs, files in os.walk(trees_dir):
            pkl_files = [x for x in files if x.endswith('trees.pkl')]
            best_file_name = sorted(pkl_files, key=self.get_iteration_number)[-1]
        return self.get_iteration_number(best_file_name)
    
    def get_literals(self):
        """
        Get all the the literals that partition the state space across actions.
        """
        # Initialize list
        all_literals = []
        # Iterate over root nodes
        for action_str, root_node in self.root_nodes.items():
                all_literals = root_node.add_literals(all_literals)
        return set(all_literals)
    
    def test_agent_epsilon_greedy(self, n_episodes=100, epsilon_test=0.05):
        """Test the agent's epsilon greedy policy and saves the results"""
        # Get best trees and load
        best_trees_path = self.get_best_trees_path()
        self.load_trees(best_trees_path)
        # Make new environment
        env = self.make_test_environment()
        # Get the best tree during training
        self.root_nodes = self.best_trees
        # Set agent exploration rate
        self.epsilon = epsilon_test
        # Initialize variables
        all_returns = []
        test_episodes = []
        # Main loop.
        for i_episode in range(n_episodes):
            # Reset environment
            episode_seed = int(self.rng.integers(30000))
            observation, info = env.reset(seed=episode_seed)
            env.action_space.seed(episode_seed)
            # Get relational state
            previous_info = self.preprocessor.get_info(observation)
            relational_state = self.get_state(
                previous_info, 
                None, 
                self.include_incomplete_states
                )
            # Episode loop
            episode_return = 0
            for t in itertools.count():
                # Act and get next relations
                action_index = self.epsilon_greedy_policy(relational_state)
                observation, reward, terminated, truncated, info = env.step(action_index)
                # Episode ends if either terminated OR truncated
                done = terminated or truncated
                info = self.preprocessor.get_info(observation)
                next_relational_state = self.get_state(
                    info, 
                    previous_info, 
                    self.include_incomplete_states
                    )
                # Reassign current to next
                relational_state = next_relational_state
                previous_info = info
                # Update return
                episode_return += reward
                # At the end of each episode
                if done:
                    # Append returns
                    all_returns.append(episode_return)
                    test_episodes.append(i_episode)
                    break

            print(round(self.epsilon, 3), i_episode, episode_return)

        # Get stats
        all_returns = np.array(all_returns)
        all_episodes = np.array(test_episodes)
        # Create files
        test_file = f"{self.save_dir}test/{self.game_name}_run_{self.run}_test.csv"
        test_data = {
            'Game': [self.game_name], 
            'Include incomplete states': [self.include_incomplete_states],
            'Run': [self.run],
            'Mean return': [np.mean(all_returns)],
            'Std Return': [np.std(all_returns)],
            'Min return': [np.amin(all_returns)],
            'Max return': [np.amax(all_returns)]
            }
        df = pd.DataFrame(test_data)
        df.to_csv(test_file)
        test_file_raw = f"{self.save_dir}test/{self.game_name}_run_{self.run}_test_raw.csv"
        test_data_raw = {
            'Game': [self.game_name] * len(all_returns), 
            'Include incomplete states': [self.include_incomplete_states] * len(all_returns), 
            'Test episode' : all_episodes,
            'Run': [self.run] * len(all_returns),
            'Returns': all_returns
            }
        df2 = pd.DataFrame(test_data_raw)
        df2.to_csv(test_file_raw)
        # Close the environment
        env.close()
        return
    
    def test_agent_deterministic(self, n_episodes=100):
        """Test the agent's greedy policy and saves the results"""
        # Get best trees and load
        best_trees_path = self.get_best_trees_path()
        self.load_trees(best_trees_path)
        # Make new environment
        env = self.make_test_environment()
        # Get the best tree during training
        self.root_nodes = self.best_trees
        # Set agent exploration rate
        self.epsilon = 0.0
        # Initialize variables
        all_returns = []
        test_episodes = []
        # Main loop.
        for i_episode in range(n_episodes):
            # Reset environment
            episode_seed = int(self.rng.integers(30000))
            observation, info = env.reset(seed=episode_seed)
            env.action_space.seed(episode_seed)
            # Get relational state
            previous_info = self.preprocessor.get_info(observation)
            relational_state = self.get_state(
                previous_info, 
                None, 
                self.include_incomplete_states
                )
            # Episode loop
            episode_return = 0
            for t in itertools.count():
                # Act and get next relations
                action_index = self.greedy_policy(relational_state)
                observation, reward, terminated, truncated, info = env.step(action_index)
                # Episode ends if either terminated OR truncated
                done = terminated or truncated
                info = self.preprocessor.get_info(observation)
                next_relational_state = self.get_state(
                    info, 
                    previous_info, 
                    self.include_incomplete_states
                    )
                # Reassign current to next
                relational_state = next_relational_state
                previous_info = info
                # Update return.
                episode_return += reward
                # At the end of each episode
                if done:
                    # Append returns
                    all_returns.append(episode_return)
                    test_episodes.append(i_episode)
                    break
            
            print(round(self.epsilon, 3), i_episode, episode_return)
        # Get stats
        all_returns = np.array(all_returns)
        test_episodes = np.array(test_episodes)
        # Create files
        test_file = f"{self.save_dir}test/{self.game_name}_run_{self.run}_test.csv"
        test_data = {
            'Game': [self.game_name], 
            'Include incomplete states': [self.include_incomplete_states],
            'Run': [self.run],
            'Iteration best model': [self.get_iteration_best_model()],
            'Mean return': [np.mean(all_returns)],
            'Std Return': [np.std(all_returns)],
            'Min return': [np.amin(all_returns)],
            'Max return': [np.amax(all_returns)]
            }
        df = pd.DataFrame(test_data)
        df.to_csv(test_file)
        test_file_raw = f"{self.save_dir}test/{self.game_name}_run_{self.run}_test_raw.csv"
        test_data_raw = {
            'Game': [self.game_name] * len(all_returns), 
            'Include incomplete states': [self.include_incomplete_states] * len(all_returns), 
            'Test episode':test_episodes,
            'Run': [self.run] * len(all_returns),
            'Returns': all_returns
            }
        df2 = pd.DataFrame(test_data_raw)
        df2.to_csv(test_file_raw)
        # Close the environment
        env.close()
        return
    
    def test_and_save_screen(self, save_imgs_dir, seed, n_episodes=1, epsilon_test=0.01, save_every=8):
        """Method for save a test game screens"""
        # Get best trees and load
        best_trees_path = self.get_best_trees_path()
        self.load_trees(best_trees_path)
        # Make new environment
        env = self.make_test_environment()
        # Get the best tree during training.
        self.root_nodes = self.best_trees
        # Set agent exploration rate.
        self.epsilon = epsilon_test
        # Initialize variables.
        i = 0
        all_returns = []
        test_games = []
        # Main loop.
        for i_episode in range(n_episodes):
            # Reset environment
            observation, info = self.env.reset(seed)
            env.action_space.seed(seed)
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
                if not relational_state:
                    action_index = self.sample_valid_action()
                    observation, reward, terminated, truncated, info = env.step(action_index)
                else:
                    # Act and get next relations
                    action_index = self.epsilon_greedy_policy(relational_state)
                    observation, reward, terminated, truncated, info = env.step(action_index)
                # Episode ends if either terminated OR truncated
                done = terminated or truncated
                if t % save_every == 0:
                    img = Image.fromarray(observation, 'RGB')
                    img.save(save_imgs_dir + f'screen-{i}.png')
                    i += 1
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
                    test_games.append(i_episode)
                    break
            
            print(round(self.epsilon, 3), i_episode, episode_return)
        # Close the environment
        env.close()
        return
