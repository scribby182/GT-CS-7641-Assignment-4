import argparse
from datetime import datetime
import logging
import random as rand
import numpy as np

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
MAX_STEPS = { 'pi': 5000, 
              'vi': 200,
              'ql': 20000, 
            }

# Configure trials per experiment
# TODO: What does this do?
NUM_TRIALS = { 'pi': 1000,
               'vi': 100,
               'ql': 1000,
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
QL_MAX_EPISODE_STEPS = 10000 # maximun steps per episode
QL_MIN_SUB_THETAS = 5  # num of consecutive episodes with little change before calling it converged
QL_ALPHAS = [0.1,]  # a list of alphas to try
QL_Q_INITS = [0,]  # a list of q-inits to try
QL_EPSILONS = [0.1]  # a list of epsilons to try
QL_EPSILON_DECAYS = [0.0001,] # a list of epsilon decays to try

# # Configure other QL experiment parameters
# QL_MAX_EPISODES = max(MAX_STEPS['ql'], NUM_TRIALS['ql'], 30000)
# QL_MIN_EPISODES = QL_MAX_EPISODES * 0.01
# QL_MAX_EPISODE_STEPS = 10000 # maximun steps per episode
# QL_MIN_SUB_THETAS = 5 # num of consecutive episodes with little change before calling it converged
# QL_ALPHAS = [0.1, 0.5, 0.9,] # a list of alphas to try
# QL_Q_INITS = ['random', 0,] # a list of q-inits to try
# QL_EPSILONS = [0.1, 0.3, 0.5,] # a list of epsilons to try
# QL_EPSILON_DECAYS = [0.0001,] # a list of epsilon decays to try


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_experiment(experiment_details, experiment, timing_key, verbose, timings, max_steps, num_trials,
                   theta=None, max_episodes=None, min_episodes=None, max_episode_steps=None,
                   min_sub_thetas=None, discount_factors=None, alphas=None, q_inits=None, epsilons=None,
                   epsilon_decays=None):

    timings[timing_key] = {}
    for details in experiment_details:
        t = datetime.now()
        logger.info("Running {} experiment: {}".format(timing_key, details.env_readable_name))
        if timing_key == 'QL':  # Q-Learning
            exp = experiment(details, verbose=verbose, max_steps=max_steps, num_trials=num_trials,
                             max_episodes=max_episodes, min_episodes=min_episodes, max_episode_steps=max_episode_steps,
                             min_sub_thetas=min_sub_thetas, theta=theta, discount_factors=discount_factors, alphas=alphas,
                             q_inits=q_inits, epsilons=epsilons, epsilon_decays=epsilon_decays)
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
        {
            'env': environments.get_racetrack_10x10( x_vel_limits=ENV_SETTINGS['racetrack']['x_vel_limits'],
                                                     y_vel_limits=ENV_SETTINGS['racetrack']['y_vel_limits'],
                                                     x_accel_limits=ENV_SETTINGS['racetrack']['x_accel_limits'],
                                                     y_accel_limits=ENV_SETTINGS['racetrack']['y_accel_limits'],
                                                     max_total_accel=ENV_SETTINGS['racetrack']['max_total_accel']),
            'name': 'racetrack_10x10',
            'readable_name': 'Racetrack (10x10)',
        },
        # {
        #     'env': environments.get_racetrack_20x10U(x_vel_limits=ENV_SETTINGS['racetrack']['x_vel_limits'],
        #                                              y_vel_limits=ENV_SETTINGS['racetrack']['y_vel_limits'],
        #                                              x_accel_limits=ENV_SETTINGS['racetrack']['x_accel_limits'],
        #                                              y_accel_limits=ENV_SETTINGS['racetrack']['y_accel_limits'],
        #                                              max_total_accel=ENV_SETTINGS['racetrack']['max_total_accel']),
        #     'name': 'racetrack_20x10u',
        #     'readable_name': 'Racetrack (20x10U)',
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

    # Run Value Iteration (VI) experiment
    if args.value or args.all:
        print('\n\n')
        run_experiment(experiment_details, experiments.ValueIterationExperiment, 'VI', verbose, timings,
                       MAX_STEPS['vi'], NUM_TRIALS['vi'], theta=VI_THETA, discount_factors=VI_DISCOUNTS)

    # Run Q-Learning (QL) experiment
    if args.ql or args.all:
        print('\n\n')
        run_experiment(experiment_details, experiments.QLearnerExperiment, 'QL', verbose, timings, MAX_STEPS['ql'],
                       NUM_TRIALS['ql'], max_episodes=QL_MAX_EPISODES, max_episode_steps=QL_MAX_EPISODE_STEPS,
                       min_episodes=QL_MIN_EPISODES, min_sub_thetas=QL_MIN_SUB_THETAS, theta=QL_THETA,
                       discount_factors=QL_DISCOUNTS, alphas=QL_ALPHAS, q_inits=QL_Q_INITS, epsilons=QL_EPSILONS,
                       epsilon_decays=QL_EPSILON_DECAYS)

    # Generate plots
    if args.plot:
        print('\n\n')
        if verbose:
            logger.info("----------")
        logger.info("Plotting results")
        plotting.plot_results(envs)

    # Output timing information
    print('\n\n')
    logger.info(timings)
    print('\n\n')

