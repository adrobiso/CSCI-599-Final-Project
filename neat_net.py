import torch
from torch.nn.functional import gumbel_softmax

from pytorch_neat.recurrent_net import RecurrentNet


def make_net(genome, config, bs):
    # TODO: get number of agents directly instead of magic number
    return RecurrentNet.create(genome, config, bs * 2)


def activate_net(net, states, debug=False, step_num=-1):
    flat_states = [state for env in states for state in env]
    outputs = net.activate(flat_states)

    outputs = outputs.split((5, 11, 3), dim=1)
    outputs = torch.cat((outputs[0], gumbel_softmax(outputs[1], hard=True), outputs[2]), dim=1)
    outputs = outputs.view(len(states), len(states[0]), -1)
    outputs = outputs.numpy()

    if debug:
        print("step: {}   outputs: {}".format(step_num, str(outputs)))

    return outputs
