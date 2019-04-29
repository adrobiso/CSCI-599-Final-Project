import pickle
import numpy as np
import matplotlib.pyplot as plt
import statistics
import string

from multi_env_multi_agent_eval import MultiEnvMultiAgentEvaluator

from make_env import make_env

from neat_net import *
# from ahneat_net import *


def view_genome(base_path, gen_num, num_runs):
    with open(base_path + 'saved_runs_saved_config', 'rb') as file:
        config = pickle.load(file)
    with open(base_path + 'saved_runs_saved_best_genome_{}'.format(gen_num), 'rb') as file:
        best = pickle.load(file)

    evaluator = MultiEnvMultiAgentEvaluator(
        make_net, activate_net, batch_size=1, make_env=make_env, max_env_steps=100
    )

    saved_fits = np.empty(num_runs)
    saved_nks = np.empty((num_runs, 10))
    for run in range(num_runs):
        fitness, nks = evaluator.view_run(best, config)
        saved_fits[run] = fitness
        saved_nks[run] = nks

    avg_fit = statistics.mean(saved_fits)
    fit_std = statistics.stdev(saved_fits)
    nk_sums = saved_nks.sum(axis=0)

    print('avg fitness: {}'.format(avg_fit))
    print('stdev: {}'.format(fit_std))

    plt.rcParams['font.size'] = 22
    plt.bar(range(10), nk_sums)
    plt.xticks(range(10), string.ascii_uppercase[0:10])
    plt.title('Utterances in {} runs of best genome in generation {}'.format(num_runs, gen_num))
    plt.xlabel('Symbol')
    plt.ylabel('Occurrences')
    plt.show()


if __name__ == "__main__":
    view_genome('saved_runs/', 50, 50)
