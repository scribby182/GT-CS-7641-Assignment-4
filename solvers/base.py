import csv
import logging

import numpy as np

from abc import ABC, abstractmethod


# Constants (default values unless provided by caller)
MAX_STEPS = 2000


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EpisodeStats(object):
    def __init__(self, num_episodes):
        self.num_episodes = num_episodes
        self.episode_lengths = np.zeros(num_episodes)
        self.episode_times = np.zeros(num_episodes)
        self.episode_rewards = np.zeros(num_episodes)
        self.episode_deltas = np.zeros(num_episodes)

    def to_csv(self, file_name):
        with open(file_name, 'w') as f:
            f.write("episode,length,time,reward,delta\n")
            writer = csv.writer(f, delimiter=',')
            writer.writerows(zip(range(self.num_episodes), self.episode_lengths, self.episode_times,
                                 self.episode_rewards, self.episode_deltas))

    @staticmethod
    def from_df(df):
        es = EpisodeStats(df.shape[0])
        es.episode_lengths = df['length'].values
        es.episode_times = df['time'].values
        es.episode_rewards = df['reward'].values
        es.episode_deltas = df['delta'].values


def one_step_lookahead(env, discount_factor, state, v):
    """
    Helper function to calculate the value for all action in a given state.

    Args:
        state: The state to consider (int)
        v: The value to use as an estimator, Vector of length env.nS

    Returns:
        A vector of length env.nA containing the expected value of each action.
    """
    try:
        action_transitions = env.P[state]
    except KeyError:
        action_transitions = env.P[env.index_to_state[state]]

    A = np.zeros(len(action_transitions))
    for a_index, transitions in enumerate(action_transitions.items()):
        for prob, next_state, reward, done in transitions[1]:
            # If required, convert next_state back to an index for the value numpy array
            if isinstance(next_state, tuple):
                next_state = env.state_to_index[next_state]
            A[a_index] += prob * (reward + discount_factor * v[next_state])
    return A


class BaseSolver(ABC):
    def __init__(self, verbose=False):
        self._verbose = verbose

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def has_converged(self):
        pass

    @abstractmethod
    def get_convergence(self):
        pass

    @abstractmethod
    def run_until_converged(self):
        pass

    @abstractmethod
    def get_environment(self):
        pass

    # Adapted from https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Policy%20Iteration%20Solution.ipynb
    def evaluate_policy(self, policy, discount_factor=1.0, max_steps=None, theta=0.00001):
        """
        Evaluate a policy given an environment and a full description of the environment's dynamics.

        Args:
            policy: The policy to evaluate
            max_steps: If not none, the number of iterations to run
            theta: We stop evaluation once our value function change is less than theta for all states.
            discount_factor: Gamma discount factor.

        Returns:
            Vector of length env.nS representing the value function.
        """
        env = self.get_environment()
# TODO: What was this for?  this block and one below it.  Couldnt understand in diff prog
#        # Get start positions
#        start = 0
#        if 'nrow' in env.__dir__():
#            # Frozen Lake
#            for s in range(env.nS):
#                row = int(s / env.nrow)
#                col = s % env.ncol
#                desc = env.desc[row][col]
#                if desc == b'S':
#                    start = s
#                    break
#        else:
#            # Cliff Walking
#            for s in range(env.nS):
#                position = np.unravel_index(s, env.shape)
#                desc = env.desc[position]
#                if desc == b'S':
#                    start = s
#                    break

        # Get values
        V = np.zeros(env.nS) # Start with a random (all 0) value function
        steps = 0
        while max_steps is None or steps < max_steps:
            delta = 0
            # For each state, perform a "full backup"
            for i_s in range(env.nS):
                v = 0
# TODO: What was this for?  this block and one below it.  Couldnt understand in diff prog
#                position = 0
#                desc = None
#                if 'nrow' in env.__dir__():
#                    # Frozen Lake
#                    row = int(s / env.nrow)
#                    col = s % env.ncol
#                    desc = env.desc[row][col]
#                else:
#                    # Cliff Walking
#                    position = np.unravel_index(s, env.shape)
#                    desc = env.desc[position]
#                if desc in b'GH':
#                    continue # terminating state...no "next" actions to evaluate
                # Look at all possible next actions
                for i_a, action_prob in enumerate(policy[i_s]):
                    # For each action, look at the possible next states...
                    try:
                        for prob, i_next_state, reward, done in env.P[i_s][i_a]:
                            # Calculate the expected value
                            # if 'nrow' not in env.__dir__() and desc == b'C':
                            #     # Cliff Walking cliff position; next state is starting position
                            #     v = V[start]
                            #     continue
                            v += action_prob * prob * (reward + discount_factor * V[i_next_state])
                    except KeyError:
                        # If we get KeyError, assume the environment indexes state and action by something other than
                        # int.  convert to qualified state and action and try again
                        qualfied_state = env.index_to_state[i_s]
                        qualified_action = env.index_to_action[i_a]
                        for prob, qualfied_next_state, reward, done in env.P[qualfied_state][qualified_action]:
                            i_next_state = env.state_to_index[qualfied_next_state]
                            # Calculate the expected value
                            #                        if 'nrow' not in env.__dir__() and desc == b'C':
                            #                            # Cliff Walking cliff position; next state is starting position
                            #                            v = V[start]
                            #                            continue
                            v += action_prob * prob * (reward + discount_factor * V[i_next_state])

                # How much the value function changed (across any states)
                delta = max(delta, np.abs(v - V[i_s]))
                V[i_s] = v
            steps += 1
            # Stop evaluating once our value function change is below a threshold (theta)
            if delta < theta:
                break

        return np.array(V)

    def render_policy(self, policy):
        env = self.get_environment()
        directions = env.directions()
        policy = np.reshape(np.argmax(policy, axis=1), env.desc.shape)

        for row in range(policy.shape[0]):
            for col in range(policy.shape[1]):
                print(directions[policy[row, col]] + ' ', end="")
            print("")

    def run_policy(self, policy, max_steps=MAX_STEPS, render_during=False):
        """
        Run once through the environment using a given policy, returning a numpy array of the rewards obtained.
        
        Side effect: Environment will be reset prior to running

        :param policy: The policy to run
        :param max_steps: The total number of steps to run. This helps prevent the agent getting "stuck"
        :param render_during: If true, render the env to stdout at each step
        :return: An ndarray of rewards for each step
        """
        policy = np.argmax(policy, axis=1)

        rewards = []

        # Clone the environment to get a fresh one
        env = self.get_environment().new_instance()
        state = env.reset()

        done = False
        steps = 0
        while not done and steps < max_steps:
            if render_during:
                env.render()

            # State can be an integer index or a tuple, depending on the environment.  Result this by catching any
            # state variables's that are tuples and convert them to state integer indices using the environment
            # This approach is taken rather than asking for permission because tuples can be interpreted as a numpy
            # index, so it is possible that a tuple state could incorrectly index the policy numpy array
            if isinstance(state, tuple):
                state = env.state_to_index[state]

            action = policy[state]
            state, reward, done, info = env.step(action)
            rewards.append(reward)
            steps += 1

        if render_during:
            env.render()

        return np.array(rewards)

    def log(self, msg, *args):
        """
        If the learner has verbose set to true, log the message with the given parameters using string.format
        :param msg: The log message
        :param args: The arguments
        :return: None
        """
        if self._verbose:
            logger.info(msg.format(*args))

