from torch import nn


class KinematicsModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super(KinematicsModel, self).__init__()

        input_size = kwargs.get('input_size', 10)
        hidden_sizes = kwargs.get('hidden_sizes', [64, 64])
        output_size = kwargs.get('output_size', 1)
        dropout_rate = kwargs.get('dropout_rate', 0.2)
        kinematics = kwargs.get('kinematics', 'inverse')

        layers = []
        for i in range(len(hidden_sizes)):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(nn.Tanh())
            layers.append(nn.BatchNorm1d(hidden_sizes[i]))
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        if kinematics == 'inverse':
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.Tanh())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
