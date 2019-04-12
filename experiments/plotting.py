import glob
import json
import logging
import matplotlib.patheffects as path_effects
import numpy as np
import os
import pandas as pd
import re
import matplotlib as mpl
mpl.use('Agg')

from os.path import basename
from matplotlib import pyplot as plt
from shutil import copyfile


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


INPUT_PATH = 'output'
OUTPUT_PATH = os.path.join('output', 'images')
REPORT_PATH = os.path.join('output', 'report')

if not os.path.exists(REPORT_PATH):
    os.makedirs(REPORT_PATH)

TO_PROCESS = {
    'PI': {
        'path': 'PI',
        'file_regex': re.compile('(.*)_grid\.csv')
    },
    'VI': {
        'path': 'VI',
        'file_regex': re.compile('(.*)_grid\.csv')
    },
    'QL': {
        'path': 'QL',
        'file_regex': re.compile('(.*)_grid\.csv')
    }
}

the_best = {}

WATERMARK = False
GATECH_USERNAME = 'DO NOT STEAL'
TERM = 'Spring 2019'


def watermark(p):
    if not WATERMARK:
        return p

    ax = plt.gca()
    for i in range(1, 11):
        p.text(0.95, 0.95 - (i * (1.0/10)), '{} {}'.format(GATECH_USERNAME, TERM), transform=ax.transAxes,
               fontsize=32, color='gray',
               ha='right', va='bottom', alpha=0.2)
    return p


def plot_episode_stats(title_base, stats, smoothing_window=50):
    # Trim the DF down based on the episode lengths
    stats = stats[stats['length'] > 0]

    # Plot the episode length over time, both as a line and histogram
    fig1 = plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.grid()
    plt.tight_layout()
    plt.plot(stats['length'])
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    plt.subplot(122)
    plt.hist(stats['length'], zorder=3)
    plt.grid(zorder=0)
    plt.xlabel("Episode Length")
    plt.ylabel("Count")
    plt.title(title_base.format("Episode Length (Histogram)"))
    fig1 = watermark(fig1)
    plt.tight_layout()

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10, 5))
    rewards_smoothed = pd.Series(stats['reward']).rolling(
        smoothing_window, min_periods=smoothing_window
    ).mean()
    plt.subplot(121)
    plt.grid()
    plt.tight_layout()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time ({})".format(smoothing_window))
    plt.subplot(122)
    plt.hist(stats['reward'], zorder=3)
    plt.grid(zorder=0)
    plt.xlabel("Episode Reward")
    plt.ylabel("Count")
    plt.title(title_base.format("Episode Reward (Histogram)"))
    fig2 = watermark(fig2)
    plt.tight_layout()

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.grid()
    plt.tight_layout()
    time_steps = np.cumsum(stats['time'])
    plt.plot(time_steps, np.arange(len(stats['time'])))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    plt.subplot(122)
    plt.hist(time_steps, zorder=3)
    plt.grid(zorder=0)
    plt.xlabel("Time Step")
    plt.ylabel("Count")
    plt.title(title_base.format("Episode Time (Histogram)"))
    fig3 = watermark(fig3)
    plt.tight_layout()

    return fig1, fig2, fig3


def plot_policy_map(title, policy, map_desc, color_map, direction_map, map_mask=None):
    """

    :param title:
    :param policy:
    :param map_desc:
    :param color_map:
    :param direction_map:
    :param map_mask: (OPTIONAL) Defines a mask in the same shape of policy that indicates which tiles should be printed.
                     Only elements that are True will have policy printed on the tile
    :return:
    """
    if map_mask is None:
        map_mask = np.ones(policy.shape, dtype=bool)

    fig = plt.figure()
    # TODO: Does this xlim/ylim even do anything?
    ax = fig.add_subplot(111)
    # FEATURE: Handle this better
    font_size = 'xx-small'
    # font_size = 'x-large'
    # if policy.shape[1] > 16 or len(direction_map[0]) > 2:
    #     font_size = 'small'
    plt.title(title)
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            y = policy.shape[0] - i - 1
            x = j
            p = plt.Rectangle((x, y), 1, 1, edgecolor='k', linewidth=0.1)
            p.set_facecolor(color_map[map_desc[i, j]])
            ax.add_patch(p)
            if map_mask[i, j]:
                text = ax.text(x+0.5, y+0.5, str(direction_map[policy[i, j]]), weight='bold', size=font_size,
                               horizontalalignment='center', verticalalignment='center', color='k')
                # TODO: Remove this?
#                text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'),
#                                       path_effects.Normal()])
    plt.axis('off')
    plt.xlim((0, policy.shape[1]))
    plt.ylim((0, policy.shape[0]))
    plt.tight_layout()

    return watermark(plt)


def plot_value_map(title, v, map_desc, color_map, map_mask=None):
    """

    :param title:
    :param v:
    :param map_desc:
    :param color_map:
    :param map_mask: (OPTIONAL) Defines a mask in the same shape of policy that indicates which tiles should be printed.
                 Only elements that are True will have policy printed on the tile
    :return:
    """
    if map_mask is None:
        map_mask = np.ones(v.shape, dtype=bool)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # FEATURE: Fix this better
    font_size = 'xx-small'
    # font_size = 'x-large'
    # if v.shape[1] > 16:
    #     font_size = 'small'

    v_min = np.min(v)
    v_max = np.max(v)
    # TODO: Disable this in more reasonble way.  Use input arg?
    # bins = np.linspace(v_min, v_max, 100)
    # v_red = np.digitize(v, bins)/100.0
    # # Flip so that numbers are red when low, not high
    # v_red = np.abs(v_red - 1)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            value = np.round(v[i, j], 1)
            if len(str(value)) > 3:
                font_size = 'xx-small'

    plt.title(title)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            y = v.shape[0] - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1, edgecolor='k', linewidth=0.1)
            p.set_facecolor(color_map[map_desc[i, j]])
            ax.add_patch(p)

            value = np.round(v[i, j], 1)

            # red = v_red[i, j]
            # if map_desc[i, j] in b'HG':
            #     continue
            if map_mask[i, j]:

                text2 = ax.text(x+0.5, y+0.5, value, size=font_size, weight='bold',
                                horizontalalignment='center', verticalalignment='center', color='k')
                # text2 = ax.text(x+0.5, y+0.5, value, size=font_size,
                #                 horizontalalignment='center', verticalalignment='center', color=(1.0, 1.0-red, 1.0-red))
                # text2.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'),
                #                        path_effects.Normal()])

    plt.axis('off')
    plt.xlim((0, v.shape[1]))
    plt.ylim((0, v.shape[0]))
    plt.tight_layout()

    return watermark(plt)


def plot_episodes(title, episodes, map_desc, color_map, direction_map,
                  path_alpha=None, fig=None):
    """
    Draw the paths of multiple episodes on a map.

    :param title:
    :param episodes:
    :param map_desc:
    :param color_map:
    :param direction_map:
    :param add_actions:
    :param add_velocity:
    :param path_alpha:
    :param fig:
    :return:
    """

    if path_alpha is None:
        path_alpha = max(1.0 / len(episodes), 0.02)

    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.get_axes()[-1]

    # Plot the background map first
    plot_map(map_desc, color_map, fig=fig)

    for episode in episodes:
        fig = plot_episode(title=title, episode=episode, map_desc=map_desc, color_map=color_map,
                           direction_map=direction_map, annotate_actions=False, annotate_velocity=False,
                           path_alpha=path_alpha, fig=fig, plot_the_map=False)
    # TODO: WATERMARK
    return fig


def plot_map(map_desc, color_map, fig=None):
    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.get_axes()[0]

    for i in range(map_desc.shape[0]):
        for j in range(map_desc.shape[1]):
            y = map_desc.shape[0] - i - 1
            x = j
            p = plt.Rectangle((x, y), 1, 1, edgecolor='k', linewidth=0.1)
            p.set_facecolor(color_map[map_desc[i, j]])
            ax.add_patch(p)
    return fig


def plot_episode(title, episode, map_desc, color_map, direction_map, annotate_actions=True, annotate_velocity=True,
                 path_alpha=1.0, path_color='r', plot_the_map=True, fig=None, annotation_fontsize=6,
                 annotation_color='r', annotation_offset=-0.5):
    """
    Plot an episode on the map, showing the path the agent has travelled during this episode.

    :param title:
    :param episode: List of (s,a,r,s') tuples describing an episode
    :param map_desc:
    :param color_map:
    :param direction_map:
    :param annotate_actions: If True, annotate all states with the action taken from them
    :param annotate_velocity: If True, plot will be annotated with velocities for each transition (note this only works
                              for an environment that has state of (x, y, vx, vy))
    :param path_color: Color used for the trace of agent's path
    :param path_alpha: Float alpha parameter passed to matplotlib when plotting the route.  If plotting multiple routes
                        to same figure, use alpha<1.0 to improve clarity (more frequently travelled routes will be
                        darker).
    :param plot_the_map: If True, plot the background map as matplotlib rectangles (if calling this function to plot
                         episodes, set this to False in all but one call to reduce plotting time
    :param fig: Optional matplotlib figure object for plotting on an existing figure (will be created if omitted)
    :param annotation_fontsize: Fontsize for all annotations
    :param annotation_color: Color for all annotations
    :param annotation_offset: Offset in x and y for all annotations (note this generally only works well as a negative)

    :return: Matplotlib figure object used
    """
    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.get_axes()[0]

    fig.suptitle(title)
    if plot_the_map:
        plot_map(map_desc, color_map, fig=fig)

    for transition in episode:
        # Transition traces go from s to s', but shift by +0.5, +0.5 because the map is plotted by the bottom left
        # corner coordinate
        x = transition[0][0] + 0.5
        y = transition[0][1] + 0.5
        x_end = transition[3][0] + 0.5
        y_end = transition[3][1] + 0.5

        # Plot the path
        ax.plot((x, x_end), (y, y_end), '-o', color=path_color, alpha=path_alpha)

        if annotate_velocity:
            # Velocity is next velocity, so pull from the s-prime
            v_xy = transition[3][2:]

            # Annotate with the velocity of a move
            arrow_xy = ((x + x_end) / 2, (y + y_end) / 2)
            annotation_xy = (arrow_xy[0] + annotation_offset, arrow_xy[1] + annotation_offset)
            ax.annotate(f'v={str(v_xy)}',
                        xy=arrow_xy,
                        xytext=annotation_xy,
                        color=annotation_color,
                        arrowprops={'arrowstyle': '->', 'color': annotation_color},
                        horizontalalignment='right',
                        verticalalignment='bottom',
                        )

        if annotate_actions:
            action = transition[1]
            arrow_xy = (x, y)
            annotation_xy = (arrow_xy[0] + annotation_offset, arrow_xy[1] + annotation_offset)
            ax.annotate(f'a={str(action)}',
                        xy=arrow_xy,
                        xytext=annotation_xy,
                        color=annotation_color,
                        arrowprops={'arrowstyle': '->', 'color': annotation_color},
                        horizontalalignment='right',
                        verticalalignment='bottom',
                        )

    ax.axis('off')
    ax.set_xlim(0, map_desc.shape[1])
    ax.set_ylim(0, map_desc.shape[0])
    fig.tight_layout()
    # TODO: WATERMARK
    return fig


def plot_time_vs_steps(title, df, xlabel="Steps", ylabel="Time (s)"):
    plt.close()
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.plot(df.index.values, df['time'], '-', linewidth=1)
    plt.legend(loc="best")
    plt.tight_layout()

    return watermark(plt)


def plot_reward_and_delta_vs_steps(title, df, xlabel="Steps", ylabel="Reward"):
    plt.close()
    plt.figure()

    f, (ax) = plt.subplots(1, 1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    lns1 = ax.plot(df.index.values, df['reward'], color='green', linewidth=1, label=ylabel)

    ex_ax = ax.twinx()
    lns2 = ex_ax.plot(df.index.values, df['delta'], color='blue', linewidth=1, label='Delta')
    ex_ax.set_ylabel('Delta')
    ex_ax.tick_params('y')

    ax.grid()
    ax.axis('tight')

    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)
    f.tight_layout()

    return watermark(plt)


# Adapted from http://code.activestate.com/recipes/578293-unicode-command-line-histograms/
def cli_hist(data, bins=10):
    bars = u' ▁▂▃▄▅▆▇█'
    n, bin_edges = np.histogram(data, bins=bins)
    n2 = map(int, np.floor(n*(len(bars)-1)/(max(n))))
    res = u' '.join(bars[i] for i in n2)

    return res


# Adapted from https://gist.github.com/joezuntz/2f3bdc2ab0ea59229907
def ascii_hist(data, bins=10):
    N, X = np.histogram(data, bins=bins)
    total = 1.0 * len(data)
    width = 50
    nmax = N.max()
    lines = []

    for (xi, n) in zip(X, N):
        bar = '#' * int(n * 1.0 * width / nmax)
        xi = '{0: <8.4g}'.format(xi).ljust(10)
        lines.append('{0}| {1}'.format(xi, bar))

    return lines


def fetch_mdp_name(file, regexp):
    search_result = regexp.search(basename(file))
    if search_result is None:
        return False, False

    mdp_name = search_result.groups()[0]

    return mdp_name, ' '.join(map(lambda x: x.capitalize(), mdp_name.split('_')))


def process_params(problem_name, params):
    param_str = '{}'.format(params['discount_factor'])
    if problem_name == 'QL':
        param_str = '{}_{}_{}_{}_{}'.format(params['alpha'], params['q_init'], params['epsilon'],
                                            params['epsilon_decay'], params['discount_factor'])

    return param_str


def find_optimal_params(problem_name, base_dir, file_regex):
    # FEATURE: Add something to catch failures here when leftover files from other runs are present.
    grid_files = glob.glob(os.path.join(base_dir, '*_grid*.csv'))
    logger.info("Grid files {}".format(grid_files))
    best_params = {}
    for f in grid_files:
        mdp, readable_mdp = fetch_mdp_name(f, file_regex)
        logger.info("MDP: {}, Readable MDP: {}".format(mdp, readable_mdp))
        df = pd.read_csv(f)
        best = df.copy()
        # Attempt to find the best params. First look at the reward mean, then median, then max. If at any point we
        # have more than one result as "best", try the next criterion
        for criterion in ['reward_mean', 'reward_median', 'reward_max']:
            best_value = np.max(best[criterion])
            best = best[best[criterion] == best_value]
            if best.shape[0] == 1:
                break

        # If we have more than one best, take the highest index.
        if best.shape[0] > 1:
            best = best.iloc[-1:]

        params = best.iloc[-1]['params']
        params = json.loads(params)
        best_index = best.iloc[-1].name

        best_params[mdp] = {
            'name': mdp,
            'readable_name': readable_mdp,
            'index': best_index,
            'params': params,
            'param_str': process_params(problem_name, params)
        }

    return best_params


def find_policy_images(base_dir, params):
    # FEATURE: This image grabber does not handle cases when velocity or other extra state variables are present.  Fix
    policy_images = {}
    for mdp in params:
        mdp_params = params[mdp]
        fileStart = os.path.join(base_dir, '{}_{}'.format(mdp_params['name'], mdp_params['param_str']))
        image_files = glob.glob(fileStart + '*.png')

        if len(image_files) == 2:
            policy_file = None
            value_file = None
            for image_file in image_files:
                if 'Value' in image_file:
                    value_file = image_file
                else:
                    policy_file = image_file

            logger.info("Value file {}, Policy File: {}".format(value_file, policy_file))
            policy_images[mdp] = {
                'value': value_file,
                'policy': policy_file
            }
        elif len(image_files) < 2:
            logger.error("Unable to find image file for {} with params {}".format(mdp, mdp_params))
        else:
            logger.warning("Found {} image files for {} with params {} ".format(len(image_files), mdp, mdp_params) + \
                        "- too many files, nothing copied")

    return policy_images


def find_data_files(base_dir, params):
    data_files = {}
    for mdp in params:
        mdp_params = params[mdp]
        files = glob.glob(os.path.join(base_dir, '{}_{}.csv'.format(mdp_params['name'], mdp_params['param_str'])))
        optimal_files = glob.glob(os.path.join(base_dir, '{}_{}_optimal.csv'.format(mdp_params['name'], mdp_params['param_str'])))
        episode_files = glob.glob(os.path.join(base_dir, '{}_{}_episode.csv'.format(mdp_params['name'], mdp_params['param_str'])))
        logger.info("files {}".format(files))
        logger.info("optimal_files {}".format(optimal_files))
        logger.info("episode_files {}".format(episode_files))
        data_files[mdp] = {
            'file': files[0],
            'optimal_file': optimal_files[0]
        }
        if len(episode_files) > 0:
            data_files[mdp]['episode_file'] = episode_files[0]

    return data_files


def copy_best_images(best_images, base_dir):
    for problem_name in best_images:
        for mdp in best_images[problem_name]:
            mdp_files = best_images[problem_name][mdp]

            dest_dir = os.path.join(base_dir, problem_name)
            policy_image = mdp_files['policy']
            value_image = mdp_files['value']

            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            policy_dest = os.path.join(dest_dir, basename(policy_image))
            value_dest = os.path.join(dest_dir, basename(value_image))
            logger.info("Copying {} to {}".format(policy_image, policy_dest))
            logger.info("Copying {} to {}".format(value_image, value_dest))

            copyfile(policy_image, policy_dest)
            copyfile(value_image, value_dest)


def copy_data_files(data_files, base_dir):
    for problem_name in data_files:
        for mdp in data_files[problem_name]:
            mdp_files = data_files[problem_name][mdp]

            dest_dir = os.path.join(base_dir, problem_name)

            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            for file_type in mdp_files:
                file_name = mdp_files[file_type]
                file_dest = os.path.join(dest_dir, basename(file_name))

                logger.info("Copying {} file from {} to {}".format(file_type, file_name, file_dest))

                copyfile(file_name, file_dest)


def plot_data(data_files, envs, base_dir):
    for problem_name in data_files:
        for mdp in data_files[problem_name]:
            env = lookup_env_from_mdp(envs, mdp)
            if env is None:
                logger.error("Unable to find env for MDP {}".format(mdp))
                return

            mdp_files = data_files[problem_name][mdp]

            step_term = 'Steps'
            if problem_name == 'QL':
                step_term = 'Episodes'

            df = pd.read_csv(mdp_files['file'])

            title = '{}: {} - Time vs {}'.format(env['readable_name'],
                                                 problem_name_to_descriptive_name(problem_name), step_term)
            file_name = os.path.join(os.path.join(base_dir, problem_name), '{}_time.png'.format(mdp))
            p = plot_time_vs_steps(title, df, xlabel=step_term)
            p = watermark(p)
            p.savefig(file_name, format='png', dpi=150)
            p.close()

            reward_term = 'Reward'
            if problem_name in ['VI', 'PI']:
                reward_term = 'Value'

            title = '{}: {} - {} and Delta vs {}'.format(env['readable_name'],
                                                         problem_name_to_descriptive_name(problem_name),
                                                         reward_term, step_term)
            file_name = os.path.join(os.path.join(base_dir, problem_name), '{}_reward_delta.png'.format(mdp))
            p = plot_reward_and_delta_vs_steps(title, df, ylabel=reward_term, xlabel=step_term)
            p = watermark(p)
            p.savefig(file_name, format='png', dpi=150)
            p.close()

            if problem_name == 'QL' and 'episode_file' in mdp_files:
                title = '{}: {} - {}'.format(env['readable_name'], problem_name_to_descriptive_name(problem_name),
                                             '{}')
                episode_df = pd.read_csv(mdp_files['episode_file'])
                q_length, q_reward, q_time = plot_episode_stats(title, episode_df)
                file_base = os.path.join(os.path.join(base_dir, problem_name), '{}_{}.png'.format(mdp, '{}'))

                logger.info("Plotting episode stats with file base {}".format(file_base))
                q_length.savefig(file_base.format('episode_length'), format='png', dpi=150)
                q_reward.savefig(file_base.format('episode_reward'), format='png', dpi=150)
                q_time.savefig(file_base.format('episode_time'), format='png', dpi=150)
                plt.close()


def lookup_env_from_mdp(envs, mdp):
    for env in envs:
        if env['name'] == mdp:
            return env

    return None


def problem_name_to_descriptive_name(problem_name):
    if problem_name == 'VI':
        return 'Value Iteration'
    if problem_name == 'PI':
        return 'Policy Iteration'
    if problem_name == 'QL':
        return "Q-Learner"
    return 'Unknown'


def plot_results(envs):
    best_params = {}
    best_images = {}
    data_files = {}
    for problem_name in TO_PROCESS:
        logger.info("Processing {}".format(problem_name))

        problem = TO_PROCESS[problem_name]
        problem_path = os.path.join(INPUT_PATH, problem['path'])
        problem_image_path = os.path.join(os.path.join(INPUT_PATH, 'images'), problem['path'])

        best_params[problem_name] = find_optimal_params(problem_name, problem_path, problem['file_regex'])
        best_images[problem_name] = find_policy_images(problem_image_path, best_params[problem_name])
        data_files[problem_name] = find_data_files(problem_path, best_params[problem_name])

    copy_best_images(best_images, REPORT_PATH)
    copy_data_files(data_files, REPORT_PATH)
    plot_data(data_files, envs, REPORT_PATH)
    params_df = pd.DataFrame(best_params)
    params_df.to_csv(os.path.join(REPORT_PATH, 'params.csv'))

