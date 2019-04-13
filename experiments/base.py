import csv
import logging
import os
import math
import pickle
import time

import numpy as np

from abc import ABC, abstractmethod

from .plotting import plot_policy_map, plot_value_map
import solvers


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants (default values unless provided by caller)
OUTPUT_DIR = 'output'
MAX_STEPS = 2000
NUM_TRIALS = 100


if not os.path.exists(os.path.join(os.getcwd(), OUTPUT_DIR)):
    os.makedirs(os.path.join(os.getcwd(), OUTPUT_DIR))
if not os.path.exists(os.path.join(os.path.join(os.getcwd(), OUTPUT_DIR), 'images')):
    os.makedirs(os.path.join(os.path.join(os.getcwd(), OUTPUT_DIR), 'images'))


class EvaluationStats(object):
    def __init__(self):
        self.rewards = list()
        self.stat_history = list()
        self.episodes = []
        self.reward_mean = 0
        self.reward_median = 0
        self.reward_std = 0
        self.reward_max = 0
        self.reward_min = 0
        self.runs = 0

    def add(self, reward, episode):
        self.rewards.append(reward)
        self.compute()
        self.episodes.append(episode)

    def compute(self):
        reward_array = np.array(self.rewards)
        self.runs = len(self.rewards)
        self.reward_mean = np.mean(reward_array)
        self.reward_median = np.median(reward_array)
        self.reward_std = np.std(reward_array)
        self.reward_max = np.max(reward_array)
        self.reward_min = np.min(reward_array)
        self.stat_history.append((
            self.reward_mean,
            self.reward_median,
            self.reward_std,
            self.reward_max,
            self.reward_min
        ))

    def to_csv(self, file_name):
        self.compute()
        means, medians, stds, maxes, mins = zip(*self.stat_history)
        with open(file_name, 'w') as f:
            f.write("step,reward,mean,median,std,max,min,episode\n")
            writer = csv.writer(f, delimiter=',')
            writer.writerows(zip(range(len(self.rewards)), self.rewards, means, medians, stds, maxes, mins,
                                 self.episodes))

    def __str__(self):
        return 'reward_mean: {}, reward_median: {}, reward_std: {}, reward_max: {}, reward_min: {}, runs: {}'.format(
            self.reward_mean,
            self.reward_median,
            self.reward_std,
            self.reward_max,
            self.reward_min,
            self.runs
        )


class ExperimentStats(object):
    def __init__(self):
        self.policies = list()
        self.vs = list()
        self.steps = list()
        self.step_times = list()
        self.rewards = list()
        self.deltas = list()
        self.converged_values = list()
        self.elapsed_time = 0
        self.optimal_policy = None

    def add(self, policy, v, step, step_time, reward, delta, converged):
        self.policies.append(policy)
        self.vs.append(v)
        self.steps.append(step)
        self.step_times.append(step_time)
        self.rewards.append(reward)
        self.deltas.append(delta)
        self.converged_values.append(converged)

    def to_csv(self, file_name):
        with open(file_name, 'w') as f:
            f.write("steps,time,reward,delta,converged\n")
            writer = csv.writer(f, delimiter=',')
            writer.writerows(zip(self.steps, self.step_times, self.rewards, self.deltas, self.converged_values))

    def pickle_results(self, file_name_base, map_shape, step_size=1, only_last=False):
        # TODO: Fix this
        print("RESULTS NOT PICKLED - NEED TO FIX THIS")
        # if only_last:
        #     policy = np.reshape(np.argmax(self.policies[-1], axis=1), map_shape)
        #     v = self.vs[-1].reshape(map_shape)
        #     file_name = file_name_base.format('Last')
        #     with open(file_name, 'wb') as f:
        #         pickle.dump({'policy': policy, 'v': v}, f)
        # else:
        #     l = len(self.policies)
        #     if step_size == 1 and l > 20:
        #         step_size = math.floor(l/20.0)
        #     for i, policy in enumerate(self.policies):
        #         if i % step_size == 0 or i == l-1:
        #             v = self.vs[i].reshape(map_shape)
        #             file_name = file_name_base.format(i)
        #             if i == l-1:
        #                 file_name = file_name_base.format('Last')
        #             with open(file_name, 'wb') as f:
        #                 pickle.dump({'policy': np.reshape(np.argmax(policy, axis=1), map_shape), 'v': v}, f)

    def plot_policies_on_map(self, file_name_base, map_desc, color_map, direction_map, experiment, step_preamble,
                             details, step_size=1, only_last=False, extra_state_variable_prefix='v=',
                             hide_terminal=True):
        """

        :param file_name_base:
        :param map_desc:
        :param color_map:
        :param direction_map:
        :param experiment:
        :param step_preamble:
        :param details:
        :param step_size:
        :param only_last:
        :param extra_state_variable_prefix: Suffix to print in front of any extra state variables printed in file names
                                            and plot titles.
        :param hide_terminal: If True, do not print policy on any terminal location.  Else, print all locations.
                              Only works if the environment has an env.is_terminal_location dictionary attribute.
        :return:
        """
        # FEATURE: Many inputs here are included in the details argument.  Roll them up, or unroll the details argument
        # and pass everything separately?
        # try:
        #     action_list = details.env.index_to_action
        # except AttributeError:
        #     action_list = [None]

        # FEATURE: Can combine only_last and other case into the same thing no?  Just make a list of those that will
        # be output and have an if for naming the last one differently
        if only_last:
            # Best policy decision for each state.  Shape as (*map_shape, all possible additional state dimensions for
            # each map location)
            policy = np.argmax(self.policies[-1], axis=1).reshape(*map_desc.shape, -1)
            v = self.vs[-1].reshape(*map_desc.shape, -1)

            # Make a state_index that holds the index corresponding to policy[i, j, [k]].  This is useful to quickly map
            # from an i,j back to a state in a few places below
            # places below.
            state_index = np.arange(0, self.policies[-1].shape[0])
            state_index = state_index.reshape(*map_desc.shape, -1)

            # If policy.shape[2] == 1 then the state space is described entirely by the map (eg: state == location on
            # map).  Otherwise, there are additional variables in state (such as velocity).  The first two variables are
            # x,y, with all other variables being something else.  Plot a separate policy plot for each of these extra
            # variable combinations.
            for i_extra_state in range(policy.shape[2]):
                if policy.shape[2] == 1:
                    policy_file_name = file_name_base.format('Policy', 'Last')
                    value_file_name = file_name_base.format('Value', 'Last')
                    title = '{}: {} - {} {}'.format(details.env_readable_name, experiment, 'Last', step_preamble)
                else:
                    # Get extra state variables
                    extra_state_variables = details.env.index_to_state[state_index[0, 0, i_extra_state]][2:]
                    policy_file_name = file_name_base.format(
                        f'Policy ({extra_state_variable_prefix}{str(extra_state_variables)})', 'Last')
                    value_file_name = file_name_base.format(
                        f'Value ({extra_state_variable_prefix}{str(extra_state_variables)})', 'Last')
                    title = '{}: {} {} - {} {}'.format(details.env_readable_name, experiment,
                                                       f'({extra_state_variable_prefix}{str(extra_state_variables)})',
                                                       'Last', step_preamble)

                # Define a mask that shows only locations that are not terminal
                mask = np.ones(policy[:, :, i_extra_state].shape)
                if hide_terminal:
                    for i in range(mask.shape[0]):
                        for j in range(mask.shape[1]):
                            # Looking for terminal location on map.  This is determined by i, j.  All k's for a
                            # given location will be the same as terminality is an incoming property to the location
                            try:
                                # Use only the first two components of state, which will be x, y
                                xy = details.env.index_to_state[state_index[i, j, 0]][:2]
                                mask[i, j] = not details.env.is_location_terminal[xy]
                            except AttributeError:
                                # env does not have an is_location_terminal attribute
                                pass

                p = plot_policy_map(title, policy[:, :, i_extra_state], map_desc, color_map, direction_map,
                                    map_mask=mask)
                p.savefig(policy_file_name, format='png', dpi=150)
                p.close()

                p = plot_value_map(title, v[:, :, i_extra_state], map_desc, color_map, map_mask=mask)
                p.savefig(value_file_name, format='png', dpi=150)
                p.close()

        else:
            l = len(self.policies)
            if step_size == 1 and l > 20:
                step_size = math.floor(l/20.0)
            for i, policy in enumerate(self.policies):
                if i % step_size == 0 or i == l-1:
                    policy = np.reshape(np.argmax(policy, axis=1), map_desc.shape)
                    v = self.vs[i].reshape(map_desc.shape)

                    file_name = file_name_base.format('Policy', i)
                    value_file_name = file_name_base.format('Value', i)
                    if i == l-1:
                        file_name = file_name_base.format('Policy', 'Last')
                        value_file_name = file_name_base.format('Value', 'Last')

                    title = '{}: {} - {} {}'.format(details.env_readable_name, experiment, step_preamble, i)

                    p = plot_policy_map(title, policy, map_desc, color_map, direction_map)
                    p.savefig(file_name, format='png', dpi=150)
                    p.close()

                    p = plot_value_map(title, v, map_desc, color_map)
                    p.savefig(value_file_name, format='png', dpi=150)
                    p.close()

    def __str__(self):
        return 'policies: {}, vs: {}, steps: {}, step_times: {}, deltas: {}, converged_values: {}'.format(
            self.policies,
            self.vs,
            self.steps,
            self.step_times,
            self.deltas,
            self.converged_values
        )


class ExperimentDetails(object):
    def __init__(self, env, env_name, env_readable_name, threads, seed):
        self.env = env
        self.env_name = env_name
        self.env_readable_name = env_readable_name
        self.threads = threads
        self.seed = seed


class BaseExperiment(ABC):
    def __init__(self, details, verbose=False, max_steps = MAX_STEPS):
        self._details = details
        self._verbose = verbose
        self._max_steps = max_steps

    @abstractmethod
    def perform(self):
        pass

    def log(self, msg, *args):
        """
        If the learner has verbose set to true, log the message with the given parameters using string.format
        :param msg: The log message
        :param args: The arguments
        :return: None
        """
        if self._verbose:
            logger.info(msg.format(*args))

    def run_solver_and_collect(self, solver, convergence_check_fn):
        stats = ExperimentStats()

        t = time.clock()
        step_count = 0
        optimal_policy = None
        best_reward = float('-inf')

        while not convergence_check_fn(solver, step_count) and step_count < self._max_steps:
            policy, v, steps, step_time, reward, delta, converged = solver.step()
            if reward > best_reward:
                best_reward = reward
                optimal_policy = policy

            # Steps returns number of steps occurred, but log this as the information for the previous step
            stats.add(policy, v, steps-1, step_time, reward, delta, converged)
            step_count += 1
        if isinstance(solver, solvers.QLearningSolver):
            self.log('Steps: {} delta: {} alpha_final: {} eps_final: {} converged: {}'.format(step_count, solver._alpha,
                                                                                              solver._epsilon, delta,
                                                                                              converged))
        else:
            self.log(
                'Steps: {} delta: {} converged: {}'.format(step_count, delta, converged))

        stats.elapsed_time = time.clock() - t
        stats.optimal_policy = stats.policies[-1]  # optimal_policy
        return stats

    def run_policy_and_collect(self, solver, policy, num_trials=NUM_TRIALS):
        """
        Run a policy multiple times, returning stats containing the mean per-episode reward for that policy

        :param solver:
        :param policy:
        :param times:
        :return:
        """
        stats = EvaluationStats()
        for i in range(num_trials):
            rewards, episode = solver.run_policy(policy)
            stats.add(reward=np.sum(rewards), episode=episode)
            # TODO: Using mean reward here seems strange.  I think I'd prefer summed rewards.  But will that break something else?
            # stats.add(np.mean(solver.run_policy(policy)))
        stats.compute()

        return stats

