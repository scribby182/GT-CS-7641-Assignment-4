import json
import os
import time
import numpy as np

from .base import BaseExperiment, OUTPUT_DIR

import solvers

PI_DIR = os.path.join(OUTPUT_DIR, 'PI')
if not os.path.exists(PI_DIR):
    os.makedirs(PI_DIR)
PKL_DIR = os.path.join(PI_DIR, 'pkl')
if not os.path.exists(PKL_DIR):
    os.makedirs(PKL_DIR)
IMG_DIR = os.path.join(os.path.join(OUTPUT_DIR, 'images'), 'PI')
if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)


class PolicyIterationExperiment(BaseExperiment):

    def __init__(self, details, verbose=False, max_steps = 2000):
        super(PolicyIterationExperiment, self).__init__(details, verbose)

    def convergence_check_fn(self, solver, step_count):
        return solver.has_converged()

    def perform(self):
        # Policy iteration
        self._details.env.reset()
        map_desc = self._details.env.unwrapped.desc

        grid_file_name = os.path.join(PI_DIR, '{}_grid.csv'.format(self._details.env_name))
        with open(grid_file_name, 'w') as f:
            f.write("params,time,steps,reward_mean,reward_median,reward_min,reward_max,reward_std\n")

        discount_factors = np.round(np.linspace(0, 0.9, num=10), 2)
        dims = len(discount_factors)
        self.log("Searching PI in {} dimensions".format(dims))

        runs = 1
        for discount_factor in discount_factors:
            t = time.clock()
            self.log("{}/{} Processing PI with discount factor {}".format(runs, dims, discount_factor))

            p = solvers.PolicyIterationSolver(self._details.env, discount_factor=discount_factor,
                                              max_policy_eval_steps=3000, verbose=self._verbose)

            stats = self.run_solver_and_collect(p, self.convergence_check_fn)

            self.log("Took {} steps".format(len(stats.steps)))
            stats.to_csv(os.path.join(PI_DIR, '{}_{}.csv'.format(self._details.env_name, discount_factor)))
            stats.pickle_results(os.path.join(PKL_DIR, '{}_{}_{}.pkl'.format(self._details.env_name, discount_factor, '{}')), map_desc.shape)
            stats.plot_policies_on_map(os.path.join(IMG_DIR, '{}_{}_{}.png'.format(self._details.env_name, discount_factor, '{}_{}')),
                                       map_desc, self._details.env.colors(), self._details.env.directions(),
                                       'Policy Iteration', 'Step', self._details, only_last=True)

            optimal_policy_stats = self.run_policy_and_collect(p, stats.optimal_policy)
            self.log('{}'.format(optimal_policy_stats))
            optimal_policy_stats.to_csv(os.path.join(PI_DIR, '{}_{}_optimal.csv'.format(self._details.env_name, discount_factor)))
            with open(grid_file_name, 'a') as f:
                f.write('"{}",{},{},{},{},{},{},{}\n'.format(
                    json.dumps({'discount_factor': discount_factor}).replace('"', '""'),
                    time.clock() - t,
                    len(optimal_policy_stats.rewards),
                    optimal_policy_stats.reward_mean,
                    optimal_policy_stats.reward_median,
                    optimal_policy_stats.reward_min,
                    optimal_policy_stats.reward_max,
                    optimal_policy_stats.reward_std,
                ))
            runs += 1

