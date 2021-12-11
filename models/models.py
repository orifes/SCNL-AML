from rtransformer.models.RTransformer import RTransformer
from torch import nn


class RTModel(nn.Module):

    def __init__(self, input_size, output_size, ksize, n_level, n, h, dropout, rnn_type='GRU'):
        super(RTModel, self).__init__()
        self.rt = RTransformer(input_size, rnn_type, ksize, n_level, n, h, dropout)
        self.linear = nn.Linear(input_size, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # x = self.encoder(x)
        output = self.rt(x)
        output = self.linear(output).double()
        return self.sig(output)


class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:, -1]))
        out = self.sig(out)
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda()
        return hidden

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:, -1]))
        out = self.sig(out)
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda()
        return hidden