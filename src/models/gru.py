import torch
import torch.nn as nn


class GRUCellStack(nn.Module):

    def __init__(self, input_size, hidden_size, n_layers):
        super().__init__()

        self.n_layers = n_layers
        layer_size = hidden_size // n_layers
        layers = [nn.GRUCell(input_size, layer_size)]
        layers.extend([nn.GRUCell(layer_size, layer_size)
                      for _ in range(n_layers-1)])
        self.layers = nn.ModuleList(layers)

    def forward(self, input, state):
        input_states = state.chunk(self.n_layers, -1)
        output_states = []
        x = input
        for i in range(self.n_layers):
            x = self.layers[i](x, input_states[i])
            output_states.append(x)
        return torch.cat(output_states, -1)
