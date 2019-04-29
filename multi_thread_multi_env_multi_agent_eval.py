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

from pytorch_neat.multi_env_eval import MultiEnvEvaluator
import numpy as np
import time

from util import dirichlet_reward


class MultiThreadMultiEnvMultiAgentEvaluator(MultiEnvEvaluator):
    def __init__(self, make_net, activate_net, make_env, batch_size=1, max_env_steps=None):
        super().__init__(make_net, activate_net, batch_size, max_env_steps, make_env, envs=[])
        self.make_env = make_env
        self.envs = None

    def eval_genome(self, genome, config, debug=False):
        envs = [self.make_env() for _ in range(self.batch_size)]
        net = self.make_net(genome, config, self.batch_size)

        fitnesses = np.zeros(self.batch_size)
        states = [env.reset() for env in envs]
        dones = [False] * self.batch_size

        nks = [0 for _ in range(envs[0].world.dim_c)]
        goal_dist = 0

        step_num = 0
        while True:
            step_num += 1
            if self.max_env_steps is not None and step_num == self.max_env_steps:
                break
            if debug:
                actions = self.activate_net(
                    net, states, debug=True, step_num=step_num)
            else:
                actions = self.activate_net(net, states)

            goal_dist = 0

            assert len(actions) == len(envs)
            for i, (env, action, done) in enumerate(zip(envs, actions, dones)):
                if not done:
                    state, reward, done, _ = env.step(action)
                    done = all(done)
                    for j in range(len(env.agents)):
                        for k, agent_k in enumerate(env.agents):
                            if j != k:
                                goal_dist -= np.sum(np.square(action[j][-3:] - agent_k.goal_b.color))
                    # get comms for dirichlet reward
                    for agent in env.agents:
                        if max(agent.action.c) == 1:
                            nks += agent.action.c
                    # all rewards the same for cooperative case TODO: competitive case
                    fitnesses[i] += reward[0]
                    if not done:
                        states[i] = state
                    dones[i] = done
            if all(dones):
                break

        d_reward = dirichlet_reward(nks)

        return (sum(fitnesses) / len(fitnesses)) + d_reward + goal_dist

    def view_run(self, genome, config, debug=False):
        envs = [self.make_env() for _ in range(self.batch_size)]
        net = self.make_net(genome, config, self.batch_size)

        fitnesses = np.zeros(self.batch_size)
        states = [env.reset() for env in envs]
        dones = [False] * self.batch_size

        step_num = 0
        while True:
            step_num += 1
            if self.max_env_steps is not None and step_num == self.max_env_steps:
                break
            if debug:
                actions = self.activate_net(
                    net, states, debug=True, step_num=step_num)
            else:
                actions = self.activate_net(net, states)
            assert len(actions) == len(envs)
            for i, (env, action, done) in enumerate(zip(envs, actions, dones)):
                if not done:
                    state, reward, done, _ = env.step(action)
                    env.render()
                    time.sleep(0.5)
                    done = all(done)
                    # all rewards the same for cooperative case TODO: competitive case
                    fitnesses[i] += reward[0]
                    if not done:
                        states[i] = state
                    dones[i] = done
            if all(dones):
                break

        return sum(fitnesses) / len(fitnesses)
