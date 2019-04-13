import argparse
from datetime import datetime
import logging
import random as rand
import numpy as np
import time

import environments
import experiments
from experiments import plotting


# Configure rewards per environment
ENV_SETTINGS = {
               'small_lake':    { 'step_prob': 0.6,
                                  'step_rew': -0.1,
                                  'hole_rew': -100,
                                  'goal_rew': 100,
                                },
               'large_lake':    { 'step_prob': 0.8,
                                  'step_rew': -0.1,
                                  'hole_rew': -100,
                                  'goal_rew': 100,
                                },
               'cliff_walking': { 'wind_prob': 0.1,
                                  'step_rew': -1,
                                  'fall_rew': -100,
                                  'goal_rew': 100,
                                },
                'racetrack':    { 'x_vel_limits': (-2, 2),
                                  'y_vel_limits': (-2, 2),
                                  'x_accel_limits': (-2, 2),
                                  'y_accel_limits': (-2, 2),
                                  'max_total_accel': 2,
                                },
              }

# Configure max steps per experiment
MAX_STEPS = {
    'pi': 5000,
    'vi': 5000,
    'ql': 20000,
    }

# Configure trials per experiment (number of times we run the optimal policy to evaluate its effectiveness)
# Is this interpretation correct???
NUM_TRIALS = {
    'pi': 100,
    'vi': 100,
    'ql': 100,
    }

# Configure thetas per experiment
PI_THETA = 0.00001
VI_THETA = 0.00001
QL_THETA = 0.001

# Configure discounts per experiment (format: iterable of discount factors)
PI_DISCOUNTS = [0.9]
VI_DISCOUNTS = [0.9]
QL_DISCOUNTS = [0.9]

# Configure other QL experiment parameters
QL_MAX_EPISODES = max(MAX_STEPS['ql'], NUM_TRIALS['ql'], 30000)
QL_MIN_EPISODES = QL_MAX_EPISODES * 0.01
QL_MAX_EPISODE_STEPS = 10000  # maximun steps per episode
QL_MIN_SUB_THETAS = 5  # num of consecutive episodes with little change before calling it converged
# List of alpha settings to try (initial, decay_rate, minimum)
QL_ALPHAS = [
    # {'initial': 0.1, 'decay': 0.001, 'min': 0.05},
    {'initial': 0.3, 'decay': 0.0005, 'min': 0.05},
    # {'initial': 0.5, 'decay': 0.001, 'min': 0.05},
]
# List of epsilon settings to try (initial, decay_rate, minimum)
QL_EPSILONS = [
    {'initial': 0.25, 'decay': 0.0005, 'min': 0.05},
    # {'initial': 0.5, 'decay': 0.001, 'min': 0.05},
    # {'initial': 0.75, 'decay': 0.001, 'min': 0.05},
]
QL_Q_INITS = [0, ]  # a list of q-inits to try (can also be 'random')


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_experiment(experiment_details, experiment, timing_key, verbose, timings, max_steps, num_trials,
                   theta=None, max_episodes=None, min_episodes=None, max_episode_steps=None,
                   min_sub_thetas=None, discount_factors=None, alphas=None, q_inits=None, epsilons=None):

    timings[timing_key] = {}
    for details in experiment_details:
        t = datetime.now()
        logger.info("Running {} experiment: {}".format(timing_key, details.env_readable_name))
        if timing_key == 'QL':  # Q-Learning
            exp = experiment(details, verbose=verbose, max_steps=max_steps, num_trials=num_trials,
                             max_episodes=max_episodes, min_episodes=min_episodes, max_episode_steps=max_episode_steps,
                             min_sub_thetas=min_sub_thetas, theta=theta, discount_factors=discount_factors, alphas=alphas,
                             q_inits=q_inits, epsilons=epsilons)
        else:  # NOT Q-Learning
            exp = experiment(details, verbose=verbose, max_steps=max_steps, num_trials=num_trials, theta=theta,
                             discount_factors=discount_factors)
        exp.perform()
        t_d = datetime.now() - t
        timings[timing_key][details.env_name] = t_d.seconds


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Run MDP experiments')
    parser.add_argument('--threads', type=int, default=-1, help='Number of threads (defaults to -1 for auto)')
    parser.add_argument('--seed', type=int, help='A random seed to set, if desired')
    parser.add_argument('--policy', action='store_true', help='Run the Policy Iteration (PI) experiment')
    parser.add_argument('--value', action='store_true', help='Run the Value Iteration (VI) experiment')
    parser.add_argument('--ql', action='store_true', help='Run the Q-Learner (QL) experiment')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--plot', action='store_true', help='Plot data results')
    parser.add_argument('--plot_paths', action='store_true', help='Plot episodic and optimal paths')
    parser.add_argument('--delete_output', action='store_true', help='Delete any previous script output')
    # TODO: There are if verbose statements below.  Can't logger handle verbosity levels?
    parser.add_argument('--verbose', action='store_true', help='If true, provide verbose output')
    args = parser.parse_args()
    verbose = args.verbose
    threads = args.threads

    # Set random seed
    seed = args.seed
    if seed is None:
        seed = np.random.randint(0, (2 ** 32) - 1)
        logger.info("Using seed {}".format(seed))
        np.random.seed(seed)
        rand.seed(seed)

    # List to make note of which solvers have run
    solvers_run = []

    # Clean up previous output
    if args.delete_output:
        logger.info("Deleting previous output")
        logger.info("----------")
        try:
            plotting.delete_output_dir()
        except FileNotFoundError:
            pass
    # Create directories required be all scripts (better way to do this?  They used to be in the initialization of
    # the imports, but that's bad and prevents us from nicely deleting here
    try:
        experiments.value_iteration.create_dirs()
        experiments.q_learner.create_dirs()
        experiments.policy_iteration.create_dirs()
        experiments.base.create_dirs()
        plotting.create_dirs()
    except PermissionError:
        # Sometimes the delete seems to linger.  Wait a few seconds then try again
        waittime = 5
        logger.warning("Deletion failed due to permission error.  Waiting {waittime} seconds then trying again")
        time.sleep(waittime)
        experiments.value_iteration.create_dirs()
        experiments.q_learner.create_dirs()
        experiments.policy_iteration.create_dirs()
        plotting.create_dirs()
        logger.info("Delete completed successfully in second attempt")

    logger.info("Creating MDPs")
    logger.info("----------")

    # Modify this list of dicts to add/remove/swap environments
    envs = [
        # {
        #     'env': environments.get_rewarding_frozen_lake_8x8_environment(ENV_SETTINGS['small_lake']['step_prob'],
        #                                                                   ENV_SETTINGS['small_lake']['step_rew'],
        #                                                                   ENV_SETTINGS['small_lake']['hole_rew'],
        #                                                                   ENV_SETTINGS['small_lake']['goal_rew']),
        #     'name': 'frozen_lake',
        #     'readable_name': 'Frozen Lake (8x8)',
        # },
        # {
        #     'env': environments.get_large_rewarding_frozen_lake_15x15_environment(ENV_SETTINGS['large_lake']['step_prob'],
        #                                                                           ENV_SETTINGS['large_lake']['step_rew'],
        #                                                                           ENV_SETTINGS['large_lake']['hole_rew'],
        #                                                                           ENV_SETTINGS['large_lake']['goal_rew']),
        #     'name': 'large_frozen_lake',
        #     'readable_name': 'Frozen Lake (15x15)',
        # },
        # {
        #     'env': environments.get_windy_cliff_walking_4x12_environment(ENV_SETTINGS['cliff_walking']['wind_prob'],
        #                                                                  ENV_SETTINGS['cliff_walking']['step_rew'],
        #                                                                  ENV_SETTINGS['cliff_walking']['fall_rew'],
        #                                                                  ENV_SETTINGS['cliff_walking']['goal_rew']),
        #     'name': 'cliff_walking',
        #     'readable_name': 'Cliff Walking (4x12)',
        # },
        # {
        #     'env': environments.get_racetrack(track='10x10',
        #                                       x_vel_limits=ENV_SETTINGS['racetrack']['x_vel_limits'],
        #                                       y_vel_limits=ENV_SETTINGS['racetrack']['y_vel_limits'],
        #                                       x_accel_limits=ENV_SETTINGS['racetrack']['x_accel_limits'],
        #                                       y_accel_limits=ENV_SETTINGS['racetrack']['y_accel_limits'],
        #                                       max_total_accel=ENV_SETTINGS['racetrack']['max_total_accel']),
        #     'name': 'racetrack_10x10',
        #     'readable_name': 'Racetrack (10x10)',
        # },
        {
            'env': environments.get_racetrack(track='10x10_basic',
                                              x_vel_limits=ENV_SETTINGS['racetrack']['x_vel_limits'],
                                              y_vel_limits=ENV_SETTINGS['racetrack']['y_vel_limits'],
                                              x_accel_limits=ENV_SETTINGS['racetrack']['x_accel_limits'],
                                              y_accel_limits=ENV_SETTINGS['racetrack']['y_accel_limits'],
                                              max_total_accel=ENV_SETTINGS['racetrack']['max_total_accel']),
            'name': 'racetrack_10x10_basic',
            'readable_name': 'Racetrack (10x10) (Basic)',
        },
        # {
        #     'env': environments.get_racetrack(track='20x20_basic',
        #                                       x_vel_limits=ENV_SETTINGS['racetrack']['x_vel_limits'],
        #                                       y_vel_limits=ENV_SETTINGS['racetrack']['y_vel_limits'],
        #                                       x_accel_limits=ENV_SETTINGS['racetrack']['x_accel_limits'],
        #                                       y_accel_limits=ENV_SETTINGS['racetrack']['y_accel_limits'],
        #                                       max_total_accel=ENV_SETTINGS['racetrack']['max_total_accel']),
        #     'name': 'racetrack_20x20_basic',
        #     'readable_name': 'Racetrack (20x20) (Basic)',
        # },
        # {
        #     'env': environments.get_racetrack(track='30x30_basic',
        #                                       x_vel_limits=ENV_SETTINGS['racetrack']['x_vel_limits'],
        #                                       y_vel_limits=ENV_SETTINGS['racetrack']['y_vel_limits'],
        #                                       x_accel_limits=ENV_SETTINGS['racetrack']['x_accel_limits'],
        #                                       y_accel_limits=ENV_SETTINGS['racetrack']['y_accel_limits'],
        #                                       max_total_accel=ENV_SETTINGS['racetrack']['max_total_accel']),
        #     'name': 'racetrack_30x30_basic',
        #     'readable_name': 'Racetrack (30x30) (Basic)',
        # },
        # {
        #     'env': environments.get_racetrack(track='10x10_oil',
        #                                       x_vel_limits=ENV_SETTINGS['racetrack']['x_vel_limits'],
        #                                       y_vel_limits=ENV_SETTINGS['racetrack']['y_vel_limits'],
        #                                       x_accel_limits=ENV_SETTINGS['racetrack']['x_accel_limits'],
        #                                       y_accel_limits=ENV_SETTINGS['racetrack']['y_accel_limits'],
        #                                       max_total_accel=ENV_SETTINGS['racetrack']['max_total_accel']),
        #     'name': 'racetrack_10x10_oil',
        #     'readable_name': 'Racetrack (10x10) (Oil)',
        # },
        # {
        #     'env': environments.get_racetrack(track='20x10_U',
        #                                       x_vel_limits=ENV_SETTINGS['racetrack']['x_vel_limits'],
        #                                       y_vel_limits=ENV_SETTINGS['racetrack']['y_vel_limits'],
        #                                       x_accel_limits=ENV_SETTINGS['racetrack']['x_accel_limits'],
        #                                       y_accel_limits=ENV_SETTINGS['racetrack']['y_accel_limits'],
        #                                       max_total_accel=ENV_SETTINGS['racetrack']['max_total_accel']),
        #     'name': 'racetrack_20x10_U',
        #     'readable_name': 'Racetrack (20x10_U)',
        # },
    ]

    # Set up experiments
    experiment_details = []
    for env in envs:
        env['env'].seed(seed)
        logger.info('{}: State space: {}, Action space: {}'.format(env['readable_name'], env['env'].unwrapped.nS,
                                                                   env['env'].unwrapped.nA))
        experiment_details.append(experiments.ExperimentDetails(
            env['env'], env['name'], env['readable_name'],
            threads=threads,
            seed=seed
        ))

    if verbose:
        logger.info("----------")
    print('\n\n')
    logger.info("Running experiments")

    timings = {} # Dict used to report experiment times (in seconds) at the end of the run

    # Run Policy Iteration (PI) experiment
    if args.policy or args.all:
        print('\n\n')
        run_experiment(experiment_details, experiments.PolicyIterationExperiment, 'PI', verbose, timings,
                       MAX_STEPS['pi'], NUM_TRIALS['pi'], theta=PI_THETA, discount_factors=PI_DISCOUNTS)
        solvers_run.append('PI')

    # Run Value Iteration (VI) experiment
    if args.value or args.all:
        print('\n\n')
        run_experiment(experiment_details, experiments.ValueIterationExperiment, 'VI', verbose, timings,
                       MAX_STEPS['vi'], NUM_TRIALS['vi'], theta=VI_THETA, discount_factors=VI_DISCOUNTS)
        solvers_run.append('VI')

    # Run Q-Learning (QL) experiment
    if args.ql or args.all:
        print('\n\n')
        run_experiment(experiment_details, experiments.QLearnerExperiment, 'QL', verbose, timings, MAX_STEPS['ql'],
                       NUM_TRIALS['ql'], max_episodes=QL_MAX_EPISODES, max_episode_steps=QL_MAX_EPISODE_STEPS,
                       min_episodes=QL_MIN_EPISODES, min_sub_thetas=QL_MIN_SUB_THETAS, theta=QL_THETA,
                       discount_factors=QL_DISCOUNTS, alphas=QL_ALPHAS, q_inits=QL_Q_INITS, epsilons=QL_EPSILONS)
        solvers_run.append('QL')

    # Generate plots
    if args.plot:
        print('\n\n')
        if verbose:
            logger.info("----------")
        logger.info("Plotting results")
        plotting.plot_results(envs)

    if args.plot_paths:
        print('\n\n')
        if verbose:
            logger.info("----------")
        logger.info("Plotting optimal and episodic paths")
        for solver_name in solvers_run:
            logger.info(f"Plotting optimal paths for {solver_name}")
            for this_experiment in experiment_details:
                plotting.plot_paths(this_experiment, solver_name=solver_name, path_type='optimal')
            if solver_name == "QL":
                logger.info(f"Plotting episodic paths for {solver_name}")
                for this_experiment in experiment_details:
                    plotting.plot_paths(this_experiment, solver_name=solver_name, path_type='episode')

    # Output timing information
    print('\n\n')
    logger.info(timings)
    print('\n\n')

