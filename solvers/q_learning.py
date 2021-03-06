import time
import numpy as np

from .base import BaseSolver, one_step_lookahead, EpisodeStats


# Adapted from https://github.com/dennybritz/reinforcement-learning/blob/master/TD/Q-Learning%20Solution.ipynb
class QLearningSolver(BaseSolver):

    def __init__(self, env, max_episodes, min_episodes, max_steps_per_episode=500, discount_factor=1.0,
                 alpha_initial=0.5, alpha_decay=False, alpha_min=0,
                 epsilon_initial=0.1, epsilon_decay=0.001, epsilon_min=0,
                 q_init=0, theta=0.0001, min_consecutive_sub_theta_episodes=10, verbose=False):

        self._env = env.unwrapped

        self._max_episodes = max_episodes
        self._min_episodes = min_episodes
        self._max_steps_per_episode = max_steps_per_episode
        self._epsilon = epsilon_initial
        self._epsilon_initial = epsilon_initial
        self._epsilon_decay = epsilon_decay
        self._epsilon_min = epsilon_min
        self._alpha = alpha_initial
        self._alpha_initial = alpha_initial
        self._alpha_decay = alpha_decay
        self._alpha_min = alpha_min
        self._discount_factor = discount_factor
        self._q_init = q_init
        self._steps = 0
        self._step_times = []
        self._last_delta = 0
        self._theta = theta
        self._stats = EpisodeStats()
        self.n_transitions = 0

        # We want to wait for a few consecutive episodes to be below theta before we consider the model converged
        self._consecutive_sub_theta_episodes = 0
        self._min_consecutive_sub_theta_episodes = min_consecutive_sub_theta_episodes

        self._init_q()

        super(QLearningSolver, self).__init__(verbose)

    def decay_alpha(self):
        if self._alpha_decay:
            self._alpha = max(self._alpha - self._alpha * self._alpha_decay, self._alpha_min)

    def decay_epsilon(self):
        if self._epsilon_decay:
            self._epsilon = max(self._epsilon - self._epsilon * self._epsilon_decay, self._epsilon_min)

    def step(self):
        """
        Perform and learn from one walk through an environment from starting state to a terminal state

        For each experience (at state s do action a, receive reward r and move to state s'), update the Q function

        :return: Tuple of:
            (
                policy,
                value function,
                total number of full walks through the environment performed so far,
                time required to compute this walk through the environment,
                sum of rewards obtained during this walk through the environment divided by total steps,
                maximum change in value of a state in this policy iteration,
                boolean denoting whether the learner is judged to be converged,
            )
        """
        # This step is a full walk through an environment, from starting state to a terminal state
        start_time = time.clock()

        # Reset the environment and pick the first action
        state = self._env.reset()

        # Record a list of all transitions
        transitions = []

        # One step in the environment
        total_reward = 0.0

        initial_epsilon = self._epsilon
        initial_alpha = self._alpha

        for t in range(self._max_steps_per_episode+1):
            # Convert a tuple state into its index, if required
            if isinstance(state, tuple):
                state = self.get_environment().state_to_index[state]

            # Take a step
            action_probs = self._policy_function(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = self._env.step(action)
            # Convert a tuple next_state into its index, if required
            if isinstance(next_state, tuple):
                next_state = self.get_environment().state_to_index[next_state]

            # TD Update
            best_next_action = np.argmax(self._Q[next_state])
            td_target = reward + self._discount_factor * self._Q[next_state, best_next_action]
            td_delta = td_target - self._Q[state, action]
            self._Q[state, action] += self._alpha * td_delta

            total_reward += reward
            self._last_delta = abs(self._alpha * td_delta)

            # Decay epsilon and alpha
            self.n_transitions += 1
            self.decay_epsilon()
            self.decay_alpha()

            # Record transition (s, a, r, s').  Try to record as fully qualified transitions (tuples), but fall back to
            # indices if possible
            try:
                this_transition = (self._env.index_to_state[state], self._env.index_to_action[action], reward, 
                                   self._env.index_to_state[next_state])
            except AttributeError:
                this_transition = (state, action, reward, next_state)
            transitions.append(this_transition)

            if done:
                break

            state = next_state

        if self._last_delta < self._theta:
            self._consecutive_sub_theta_episodes += 1
        else:
            self._consecutive_sub_theta_episodes = 0

        run_time = time.clock() - start_time
        # Update statistics for this episode
        self._stats.add(episode_length=t + 1, episode_time=run_time, episode_reward=total_reward,
                        episode_delta=self._last_delta, episode_transitions=transitions,
                        episode_start_epsilon=initial_epsilon, episode_end_epsilon=self._epsilon,
                        episode_start_alpha=initial_alpha, episode_end_alpha=self._alpha)

        self._step_times.append(run_time)

        self._steps += 1

        return self.get_policy(), self.get_value(), self._steps, self._step_times[-1], \
            total_reward, self._last_delta, self.has_converged()

    def reset(self):
        self._init_q()
        self._steps = 0
        self._step_times = []
        self._last_delta = 0
        self._epsilon = self._epsilon_initial
        self._stats = EpisodeStats()
        self._consecutive_sub_theta_episodes = 0

    def has_converged(self):
        return (self._steps >= self._min_episodes and
                self._consecutive_sub_theta_episodes >= self._min_consecutive_sub_theta_episodes) \
               or self._steps > self._max_episodes

    def get_convergence(self):
        return self._last_delta

    def run_until_converged(self):
        while not self.has_converged():
            self.step()

    def get_environment(self):
        return self._env

    def get_stats(self):
        return self._stats

    def get_q(self):
        return self._Q

    def get_policy(self):
        policy = np.zeros([self._env.nS, self._env.nA])
        for s in range(self._env.nS):
            best_action = np.argmax(self._Q[s])
            # Always take the best action
            policy[s, best_action] = 1.0

        return policy

    def get_value(self):
        v = np.zeros(self._env.nS)
        for s in range(self._env.nS):
            v[s] = np.max(self._Q[s])

        return v

    def _init_q(self):
        if self._q_init == 'random':
            self._Q = np.random.rand(self._env.observation_space.n, self._env.action_space.n)/1000.0
        elif int(self._q_init) == 0:
            self._Q = np.zeros(shape=(self._env.observation_space.n, self._env.action_space.n))
        else:
            self._Q = np.full((self._env.observation_space.n, self._env.action_space.n), float(self._q_init))

    def _policy_function(self, observation):
        A = np.ones(self._env.action_space.n, dtype=float) * self._epsilon / self._env.action_space.n
        # Convert observation (state) to index for state, if necessary
        if isinstance(observation, tuple):
            observation = self.get_environment().state_to_index[observation]
        best_action = np.argmax(self._Q[observation])
        A[best_action] += (1.0 - self._epsilon)
        return A

    # Adapted from https://github.com/dennybritz/reinforcement-learning/blob/master/TD/Q-Learning%20Solution.ipynb
    def _make_epsilon_greedy_policy(self):
        """
        Creates an epsilon-greedy policy based on a given Q-function and epsilon.

        Args:
            Q: A dictionary that maps from state -> action-values.
                Each value is a numpy array of length nA (see below)
            epsilon: The probability to select a random action . float between 0 and 1.
            nA: Number of actions in the environment.

        Returns:
            A function that takes the observation as an argument and returns
            the probabilities for each action in the form of a numpy array of length nA.
        """
        def policy_fn(observation):
            A = np.ones(self._env.action_space.n, dtype=float) * self._epsilon / self._env.action_space.n
            best_action = np.argmax(self._Q[observation])
            A[best_action] += (1.0 - self._epsilon)
            return A
        return policy_fn

