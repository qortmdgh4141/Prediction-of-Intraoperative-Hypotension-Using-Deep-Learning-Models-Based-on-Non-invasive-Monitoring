import torch

from torch import nn
from torch.nn import functional as F
from ml_component.model.galr import Segment1d, GALR

class XGALR(nn.Module):
    def __init__(self,
                 num_features: int, embedding_linear_bias: bool, num_heads: int, batch_first: bool, num_classes: int,
                 multihead_embedding_dim: int, multihead_sequences: int, multihead_axis: int,
                 galr_embedding_dim: int, galr_axis: int,
                 galr_chunk_size: int, galr_hop_size: int,
                 galr_hidden_channels: int, galr_num_blocks: int, galr_bidirectional: bool, galr_eps: float, galr_dropout: float,
                 temperature: float, save_attn: bool):
        super(XGALR, self).__init__()
        self.name = self.__class__.__name__
        self.temperature = temperature
        self.save_attn = save_attn

        # INFO: Multi-head attention layer.
        self.embedding = Embedding(num_features=num_features, embedding_dim=multihead_embedding_dim, axis=multihead_axis, embedding_linear_bias=embedding_linear_bias, multihead_sequences=multihead_sequences)
        self.bn = nn.BatchNorm1d(num_features)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=multihead_embedding_dim, num_heads=num_heads, batch_first=batch_first)

        # INFO: GALR layers
        self.gembedding = Embedding(num_features=num_features, embedding_dim=galr_embedding_dim, axis=galr_axis, embedding_linear_bias=embedding_linear_bias)
        self.galr = nn.Sequential(
            Segment1d(chunk_size=galr_chunk_size, hop_size=galr_hop_size),
            GALR(num_blocks=galr_num_blocks,
                 num_features=num_features, hidden_channels=galr_hidden_channels, batch_first=batch_first, bidirectional=galr_bidirectional,
                 num_heads=num_heads, eps=galr_eps, dropout=galr_dropout)
        )

        # TRY: Change this layer to FC
        self.conv1d = nn.Conv1d(num_features, 1, kernel_size=1, stride=1)

        # TODO: Implement suitable FC layer in the future.
        self.fc = nn.Sequential(
            nn.Linear(multihead_sequences, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_x):
        batch, sequences, features = input_x.size()
        zero_padding = torch.zeros(batch, sequences, 1)
        #zero_padding = zero_padding.to('cuda')
        zero_padding = zero_padding.to(input_x.device)
        input_x = torch.cat([input_x, zero_padding], dim=2)
        features = 8

        # INFO: Multi-head attention embedding inputs
        embedded_out = self.embedding(input_x)      # embedded output for multihead_attention
        embedded_out = self.bn(embedded_out)
        mthd_out, attention_weights = self.multihead_attention(embedded_out, embedded_out, embedded_out)

        # INFO: GALR embedding inputs
        embedded_out = self.gembedding(input_x)     # embedded output for galr block
        galr_out, gatt = self.galr(embedded_out)
        galr_out = galr_out.view(batch, features, -1)

        # INFO Temperature value
        out = torch.bmm(self.temperature * attention_weights, galr_out)
        out = F.relu(self.conv1d(out))
        out = out.view(batch, -1)
        out = self.fc(out)

        if self.save_attn:
            out = (out, attention_weights, gatt)

        return out

class Embedding(nn.Module):
    """Linear embedding module."""
    def __init__(self, num_features: int, embedding_dim: int, axis: int, embedding_linear_bias: bool, multihead_sequences=3000):
        super(Embedding, self).__init__()
        self.layers = []
        self.axis = axis

        def create_linear(in_c, out_c):
            layer = nn.Sequential(nn.Linear(in_features=in_c, out_features=out_c, bias=embedding_linear_bias))
            return layer

        # multihead_linear
        if self.axis == 1:
            self.layers = nn.ModuleList([create_linear(multihead_sequences, embedding_dim) for _ in range(num_features)])
        # galr_linear
        else:
            self.layers = create_linear(num_features, embedding_dim)

    def forward(self, x):
        # multihead_foward
        if self.axis == 1:
            # mac is dropped (7 features) , so feature embedding is prerequisite for multihead attention dimension
            x = x.transpose(1, 2).contiguous()             # (batch, sequence, features) -> (batch, features, sequences)
            output_list = []
            for idx in range(x.shape[1]):
                _x = x[:, idx, :]
                _x = _x.reshape(-1, 1, _x.shape[-1])        # (batches ,1, sequences)
                output_list.append(self.layers[idx](_x))    # (batches, idx, 1, embedding_dim)
            out = torch.concat(output_list, dim=1)          # (batches, num_of_features, embedding_dim)
        # galr_foward
        else:
            out = self.layers(x)
            out = out.transpose(1, 2).contiguous()  # (batches, sequence, embedding_dim)

        return out