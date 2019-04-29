# Copyright (c) 2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import multiprocessing
import os
import pickle

import click
import neat

from multi_env_multi_agent_eval import MultiEnvMultiAgentEvaluator
from multi_thread_multi_env_multi_agent_eval import MultiThreadMultiEnvMultiAgentEvaluator

from pytorch_neat.neat_reporter import LogReporter

from make_env import make_env

from neat_net import *
# from ahneat_net import *
cfg = "neat.cfg"

@click.command()
@click.option("--n_generations", type=int, default=1000)
@click.option("--n_processes", type=int, default=2)
def run(n_generations, n_processes):
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    config_path = os.path.join(os.path.dirname(__file__), cfg)
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    evaluator = MultiThreadMultiEnvMultiAgentEvaluator(
        make_net, activate_net, batch_size=20, make_env=make_env, max_env_steps=100
    )

    if n_processes > 1:
        pool = multiprocessing.Pool(processes=n_processes)

        def eval_genomes(genomes, config):
            fitnesses = pool.starmap(
                evaluator.eval_genome, ((genome, config) for _, genome in genomes)
            )
            for (_, genome), fitness in zip(genomes, fitnesses):
                genome.fitness = fitness

    else:
        def eval_genomes(genomes, config):
            for _, genome in genomes:
                genome.fitness = evaluator.eval_genome(genome, config, debug=False)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    reporter = neat.StdOutReporter(True)
    pop.add_reporter(reporter)
    logger = LogReporter("saved_runs/neat.log", evaluator.eval_genome)
    pop.add_reporter(logger)

    with open('saved_runs/saved_config', 'wb') as file:
        pickle.dump(config, file)
    with open('saved_runs/saved_pop_0', 'wb') as file:
        pickle.dump((pop.population, pop.generation), file)

    # manual initialization of pop.species since reporters can't be pickled
    # pop = neat.Population(config, (population, None, generation))
    # pop.species = config.species_set_type(config.species_set_config, pop.reporters)
    # pop.species.speciate(config, pop.population, pop.generation)

    best = None
    save_interval = int(n_generations / 20)
    n = 0
    while n < n_generations:
        best = pop.run(eval_genomes, save_interval)
        n += save_interval

        with open('saved_runs/saved_pop_{}'.format(n), 'wb') as file:
            pickle.dump((pop.population, pop.generation), file)
        with open('saved_runs/saved_best_genome_{}'.format(n), 'wb') as file:
            pickle.dump(best, file)


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
