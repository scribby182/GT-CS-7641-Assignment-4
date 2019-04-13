import json
import os
import time
import numpy as np

from .base import BaseExperiment, OUTPUT_DIR
from experiments.plotting import params_to_filename_base
import solvers


# Constants (default values unless provided by caller)
MAX_STEPS = 2000
NUM_TRIALS = 100
MAX_EPISODES = 2000
MIN_EPISODES = MAX_EPISODES * 0.05
MAX_EPISODE_STEPS = 500
ALPHAS = [
    {'initial': 0.5, 'decay': 0.0001, 'min': 0.05},
]
Q_INITS = ['random', 0]
EPSILONS = [
    {'initial': 0.5, 'decay': 0.0001, 'min': 0.05},
]
DISCOUNT_MIN = 0
DISCOUNT_MAX = 0.9
NUM_DISCOUNTS = 10
MIN_SUB_THETAS = 5
THETA = 0.0001

QL_DIR = os.path.join(OUTPUT_DIR, 'QL')
PKL_DIR = os.path.join(QL_DIR, 'pkl')
IMG_DIR = os.path.join(os.path.join(OUTPUT_DIR, 'images'), 'QL')


def create_dirs():
    if not os.path.exists(QL_DIR):
        os.makedirs(QL_DIR)
    if not os.path.exists(PKL_DIR):
        os.makedirs(PKL_DIR)
    if not os.path.exists(IMG_DIR):
        os.makedirs(IMG_DIR)


class QLearnerExperiment(BaseExperiment):

    def __init__(self, details, verbose=False, max_steps=MAX_STEPS, num_trials=NUM_TRIALS,
                 max_episodes=MAX_EPISODES, min_episodes=MIN_EPISODES, max_episode_steps=MAX_EPISODE_STEPS,
                 min_sub_thetas=MIN_SUB_THETAS, theta=THETA, discount_factors=None,
                 alphas=None, q_inits=None, epsilons=None):
        if alphas is None:
            alphas = ALPHAS
        if q_inits is None:
            q_inits = Q_INITS
        if epsilons is None:
            epsilons = EPSILONS

        if discount_factors is None:
            discount_factors = np.round(np.linspace(DISCOUNT_MIN, DISCOUNT_MAX, NUM_DISCOUNTS), 2)
        self._max_episodes = max_episodes
        self._max_episode_steps = max_episode_steps
        self._min_episodes = min_episodes
        self._num_trials = num_trials
        self._min_sub_thetas = min_sub_thetas
        self._theta = theta
        self._discount_factors = discount_factors
        self._alphas = alphas
        self._q_inits = q_inits
        self._epsilons = epsilons

        super(QLearnerExperiment, self).__init__(details, verbose, max_steps)

    def convergence_check_fn(self, solver, step_count):
        return solver.has_converged()

    def perform(self):
        # Q-Learner
        self._details.env.reset()
        map_desc = self._details.env.unwrapped.desc

        grid_file_name = os.path.join(QL_DIR, '{}_grid.csv'.format(self._details.env_name))
        with open(grid_file_name, 'w') as f:
            f.write("params,q_init,alpha_initial,alpha_min,alpha_decay,epsilon_initial,epsilon_min,epsilon_decay,"
                    "discount_factor,time,steps,reward_mean,reward_median,reward_min,reward_max,reward_std\n")

        dims = len(self._discount_factors) * len(self._alphas) * len(self._q_inits) * len(self._epsilons)
        self.log("Searching Q in {} dimensions".format(dims))

        runs = 1
        for alpha in self._alphas:
            for q_init in self._q_inits:
                for epsilon in self._epsilons:
                    for discount_factor in self._discount_factors:
                        t = time.clock()
                        self.log(f"{runs}/{dims} Processing QL with alpha {alpha['initial']}->{alpha['min']} "
                                 f"(decay={alpha['decay']}), q_init {q_init}, epsilon {epsilon['initial']}->"
                                 f"{epsilon['min']} (decay={epsilon['decay']}), discount_factor {discount_factor}")

                        # Build a QLeaningSolver object
                        qs = solvers.QLearningSolver(self._details.env, self._max_episodes, self._min_episodes,
                                                     max_steps_per_episode=self._max_episode_steps,
                                                     discount_factor=discount_factor,
                                                     alpha_initial=alpha['initial'], alpha_decay=alpha['decay'],
                                                     alpha_min=alpha['min'], epsilon_initial=epsilon['initial'],
                                                     epsilon_decay=epsilon['decay'], epsilon_min=epsilon['min'],
                                                     q_init=q_init,
                                                     min_consecutive_sub_theta_episodes=self._min_sub_thetas,
                                                     verbose=self._verbose, theta=self._theta)

                        # Run the solver to generate an optimal policy.  Stats object contains details about all
                        # optimal policy and
                        # s
                        stats = self.run_solver_and_collect(qs, self.convergence_check_fn)

                        self.log("Took {} episodes".format(len(stats.steps)))
                            
                        filename_base = params_to_filename_base(self._details.env_name,
                                                            alpha["initial"], alpha["min"], alpha["decay"], q_init, 
                                                            epsilon["initial"], epsilon["min"], epsilon["decay"], 
                                                            discount_factor)

                        stats.to_csv(os.path.join(QL_DIR, f'{filename_base}.csv'))
                        stats.pickle_results(os.path.join(PKL_DIR, f'{filename_base}_{{}}.pkl'), map_desc.shape,
                                             step_size=self._max_episodes / 20.0)
                        stats.plot_policies_on_map(os.path.join(IMG_DIR, f'{filename_base}_{{}}_{{}}.png'),
                                                   map_desc, self._details.env.colors(),
                                                   self._details.env.directions(),
                                                   'Q-Learner', 'Episode', self._details,
                                                   step_size=self._max_episodes / 20.0,
                                                   only_last=True)

                        # We have extra stats about the episode we might want to look at later
                        episode_stats = qs.get_stats()
                        episode_stats.to_csv(os.path.join(QL_DIR, f'{filename_base}_episode.csv'))

                        optimal_policy_stats = self.run_policy_and_collect(qs, stats.best_policy, self._num_trials)
                        self.log('{}'.format(optimal_policy_stats))
                        optimal_policy_stats.to_csv(os.path.join(QL_DIR, f'{filename_base}_optimal.csv'))

                        with open(grid_file_name, 'a') as f:
                            # Data as an iterable of numbers and such
                            # TODO: Replace these instances where headers are above and numbers written down here with
                            # a csv or pandas to csv call?
                            # Single group version (for legacy support)
                            params = json.dumps({
                                    'q_init': q_init,
                                    'alpha_initial': alpha['initial'],
                                    'alpha_min': alpha['min'],
                                    'alpha_decay': alpha['decay'],
                                    'epsilon_initial': epsilon['initial'],
                                    'epsilon_min': epsilon['min'],
                                    'epsilon_decay': epsilon['decay'],
                                    'discount_factor': discount_factor,
                                }).replace('"', '""')
                            data = [
                                f'"{params}"', q_init, alpha['initial'], alpha['min'], alpha['decay'], epsilon['initial'],
                                epsilon['min'], epsilon['decay'], discount_factor, time.clock() - t,
                                len(optimal_policy_stats.rewards), optimal_policy_stats.reward_mean,
                                optimal_policy_stats.reward_median, optimal_policy_stats.reward_min,
                                optimal_policy_stats.reward_max, optimal_policy_stats.reward_std,
                            ]
                            # Convert to a single csv string
                            data_as_string = ",".join([str(d) for d in data])
                            f.write(f'{data_as_string}\n')
                        runs += 1
