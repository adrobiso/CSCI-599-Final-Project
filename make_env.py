from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios


def make_env():
    return make_multiagent_env("simple_reference")


def make_multiagent_env(scenario_name, benchmark=False):
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env
