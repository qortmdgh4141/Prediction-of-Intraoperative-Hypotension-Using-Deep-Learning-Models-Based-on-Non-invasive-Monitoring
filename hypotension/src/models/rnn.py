import torch
import torch.nn.functional as F
from torch import nn


class ValinaLSTM(nn.Module):
    def __init__(self, **kwargs):
        super(ValinaLSTM, self).__init__()
        self.name = self.__class__.__name__
        self.layers = kwargs['layers']
        self.hidden_units = kwargs['hidden_units']
        self.lstm = nn.LSTM(input_size=kwargs['features'],
                            hidden_size=kwargs['hidden_units'],
                            num_layers=kwargs['layers'],
                            bidirectional=kwargs['bidirectional'])
        self.linear = nn.Linear(kwargs['hidden_units'], 128)
        self.fc = nn.Linear(128, kwargs['number_of_classes'])

    def forward(self, X, hidden):
        X = X.transpose(0, 1)
        outputs, hidden = self.lstm(X, hidden)
        outputs = outputs[-1]
        out = self.linear(outputs)
        out = F.relu(out)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size, device):
        # hidden = torch.zeros(2, self.hidden_units, dtype=torch.float64)
        hidden = (torch.ones(self.layers, batch_size, self.hidden_units, device=device, dtype=torch.float64),
                  torch.ones(self.layers, batch_size, self.hidden_units, device=device, dtype=torch.float64))
        return hidden
