import torch
from torch.nn.functional import gumbel_softmax

from pytorch_neat.activations import tanh_activation
from pytorch_neat.adaptive_linear_net import AdaptiveLinearNet


def make_net(genome, config, batch_size):
    input_coords = [[x / 20, 1.0] for x in range(21)]
    output_coords = [[x / 18, -1.0] for x in range(19)]
    return AdaptiveLinearNet.create(
        genome,
        config,
        input_coords=input_coords,
        output_coords=output_coords,
        weight_threshold=0.4,
        batch_size=batch_size*2,
        activation=tanh_activation,
        output_activation=tanh_activation,
        device="cpu",
    )


def activate_net(net, states, debug=False, step_num=0):
    flat_states = [state for env in states for state in env]

    if debug and step_num == 1:
        print("\n" + "=" * 20 + " DEBUG " + "=" * 20)
        print(net.delta_w_node)
        print("W init: ", net.input_to_output[0])

    outputs = net.activate(flat_states)
    outputs = outputs.split((5, 11, 3), dim=1)
    outputs = torch.cat((outputs[0], gumbel_softmax(outputs[1], hard=True), outputs[2]), dim=1)
    outputs = outputs.view(len(states), len(states[0]), -1)
    outputs = outputs.numpy()

    if debug and (step_num - 1) % 100 == 0:
        print("\nStep {}".format(step_num - 1))
        print("Outputs: ", outputs[0])
        print("Delta W: ", net.delta_w[0])
        print("W: ", net.input_to_output[0])

    return outputs
