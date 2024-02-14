import torch
import torch.nn.functional as F
import copy
import math
from torch import nn, ones
# from src.models.attentions import MultiHeadAttention

EPS = 1e-12

class GALR_best_acc(nn.Module):
    def __init__(self, num_features, hidden_channels, block_num = 1, num_heads=8,  norm=True, dropout=1e-1 , eps=EPS, num_of_classes = 2, **kwargs):
        super().__init__()

        self.block_num = block_num

        self.fc = nn.Sequential(
            nn.Linear(3000*8, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_of_classes)
        )

        self.intra_chunk_block = IntraChunkRNN(num_features, hidden_channels=hidden_channels, norm=norm, eps=eps)
        self.intra_chunk_block1 = IntraChunkRNN(num_features, hidden_channels=hidden_channels,first=True, norm=norm, eps=eps)


        chunk_size = 100
        down_chunk_size = 100
        self.inter_chunk_block = LowDimensionGloballyAttentiveBlock(num_features, chunk_size=chunk_size, down_chunk_size = down_chunk_size, num_heads=num_heads, norm=norm, dropout=dropout, eps=eps)
        self.inter_chunk_block1 = LowDimensionGloballyAttentiveBlock(num_features, chunk_size=chunk_size,
                                                                    down_chunk_size=down_chunk_size,
                                                                    num_heads=num_heads, last=True, norm=norm,
                                                                    dropout=dropout, eps=eps)

    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, S, chunk_size)
        Returns:
            output (batch_size, num_features, S, chunk_size)
        """
        output = self.intra_chunk_block1(input)
        output = self.inter_chunk_block(output)

        for i in range(self.block_num):
            output = self.intra_chunk_block(output)
            output = self.inter_chunk_block(output)

        output = self.intra_chunk_block(output)
        output = self.inter_chunk_block1(output)

        output = self.fc(output)

        return output



class GloballyAttentiveBlockBase(nn.Module):
    def __init__(self):
        super().__init__()

    def positional_encoding(self, length: int, dimension: int, base=10000):
        """
        Args:
            length <int>:
            dimension <int>:
        Returns:
            output (length, dimension): positional encording
        """
        assert dimension % 2 == 0, "dimension is expected even number but given odd number."

        position = torch.arange(length) # (length,)
        position = position.unsqueeze(dim=1) # (length, 1)
        index = torch.arange(dimension//2) / dimension # (dimension // 2,)
        index = index.unsqueeze(dim=0) # (1, dimension // 2)
        indices = position / base**index
        output = torch.cat([torch.sin(indices), torch.cos(indices)], dim=1)

        return output


class LowDimensionGloballyAttentiveBlock(GloballyAttentiveBlockBase):
    def __init__(self, num_features, chunk_size=100, down_chunk_size=100, num_heads=8, last=False, norm=True,
                 dropout=1e-1, eps=EPS):
        super().__init__()

        #self.fc = nn.Sequential(nn.Linear(3000 * 8, 512),nn.ReLU())## remove

        self.last = last
        self.down_chunk_size = down_chunk_size
        self.norm = norm

        self.fc_map = nn.Linear(chunk_size, down_chunk_size)

        if self.norm:
            self.norm2d_in = LayerNormAlongChannel(num_features, eps=eps)

        self.multihead_attn = nn.MultiheadAttention(num_features, num_heads)

        if dropout is not None:
            self.dropout = True
            self.dropout1d = nn.Dropout(p=dropout)
        else:
            self.dropout = False

        #if self.norm:
        #    norm_name = 'cLN' if causal else 'gLN'
        #    self.norm2d_out = choose_layer_norm(norm_name, num_features, causal=causal, eps=eps)
        if self.norm:
            self.norm2d_out = GlobalLayerNorm(num_features, eps=eps)


    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, S, K): K is chunk size
        Returns:
            output (batch_size, num_features, S, K)
        """
        Q = self.down_chunk_size
        batch_size, num_features, S, K = input.size()

        x = self.fc_map(input)  # (batch_size, num_features, S, K) -> (batch_size, num_features, S, Q)
        input = x

        if self.norm:
            x = self.norm2d_in(x)  # -> (batch_size, num_features, S, Q)

        encoding = self.positional_encoding(length=S * Q, dimension=num_features).permute(1, 0).view(num_features, S,
                                                                                                     Q).to(x.device)
        x = x + encoding  # -> (batch_size, num_features, S, Q)
        x = x.permute(2, 0, 3, 1).contiguous()  # -> (S, batch_size, Q, num_features)
        x = x.view(S, batch_size * Q, num_features)  # -> (S, batch_size*Q, num_features)

        residual = x  # (S, batch_size*Q, num_features)
        x, _ = self.multihead_attn(x, x,
                                   x)  # (T_tgt, batch_size, num_features), (batch_size, T_tgt, T_src), where T_tgt = T_src = T

        if self.dropout:
            x = self.dropout1d(x)
        x = x + residual  # -> (S, batch_size*Q, num_features)
        x = x.view(S, batch_size, Q, num_features)
        x = x.permute(1, 3, 0, 2).contiguous()  # -> (batch_size, num_features, S, Q)

        if self.norm:
            x = self.norm2d_out(x)  # -> (batch_size, num_features, S, Q)

        #x = self.fc_inv(x)  # (batch_size, num_features, S, Q) -> (batch_size, num_features, S, K)
        x = x + input
        output = x.view(batch_size, num_features, S, K)

        if self.last:
            output = x.view(batch_size,-1)
        #output = self.fc(output)

        return output

class IntraChunkRNN(nn.Module):
    def __init__(self, num_features , hidden_channels, norm=True, first=False, eps=EPS):##delete type
        super().__init__()

        self.first = first
        self.num_features, self.hidden_channels = num_features, hidden_channels
        num_directions = 1  # bi-direction
        self.norm = norm

        self.rnn = nn.LSTM(input_size=num_features, hidden_size=hidden_channels, batch_first=True, bidirectional=False)

        self.fc = nn.Linear(num_directions * hidden_channels, num_features)

        #if self.norm:
        #    norm_name = 'gLN'
        #    self.norm1d = choose_layer_norm(norm_name, num_features, causal=False, eps=eps)
        self.norm1d = GlobalLayerNorm(num_features, eps=eps)

    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, S, chunk_size)
        Returns:
            output (batch_size, num_features, S, chunk_size)
        """
        num_features = self.num_features
        if self.first:
            batch_size, totalseq, num_features = input.size()
            input.permute(0,2,1)
        else:
            batch_size, num_features, S, K = input.size()

        input = input.view(batch_size,num_features,30,100)
        batch_size, _, S, chunk_size = input.size()

        self.rnn.flatten_parameters()

        residual = input  # (batch_size, num_features, S, chunk_size)
        x = input.permute(0, 2, 3, 1).contiguous()  # -> (batch_size, S, chunk_size, num_features)
        x = x.view(batch_size * S, chunk_size, num_features)
        x, _ = self.rnn(
            x)  # (batch_size*S, chunk_size, num_features) -> (batch_size*S, chunk_size, num_directions*hidden_channels)
        x = self.fc(x)  # -> (batch_size*S, chunk_size, num_features)
        x = x.view(batch_size, S * chunk_size, num_features)  # (batch_size, S*chunk_size, num_features)
        x = x.permute(0, 2, 1).contiguous()  # -> (batch_size, num_features, S*chunk_size)
        if self.norm:
            x = self.norm1d(x)  # (batch_size, num_features, S*chunk_size)
        x = x.view(batch_size, num_features, S, chunk_size)  # -> (batch_size, num_features, S, chunk_size)
        output = x + residual

        return output


class LayerNormAlongChannel(nn.Module):
    def __init__(self, num_features, eps=EPS):
        super().__init__()

        self.num_features = num_features
        self.eps = eps

        self.norm = nn.LayerNorm(num_features, eps=eps)

    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, *)
        Returns:
            output (batch_size, num_features, *)
        """
        n_dims = input.dim()
        dims = list(range(n_dims))
        permuted_dims = dims[0:1] + dims[2:] + dims[1:2]
        x = input.permute(*permuted_dims)
        x = self.norm(x)
        permuted_dims = dims[0:1] + dims[-1:] + dims[1:-1]
        output = x.permute(*permuted_dims).contiguous()

        return output

##GlobalLayerNorm(num_features, eps=eps)
#directly use GlobalLayerNorm

class GlobalLayerNorm(nn.Module):
    def __init__(self, num_features, eps=EPS):
        super().__init__()

        self.num_features = num_features
        self.eps = eps

        self.norm = nn.GroupNorm(1, num_features, eps=eps)

    def forward(self, input):
        """
        Args:
            input (batch_size, C, *)
        Returns:
            output (batch_size, C, *)
        """
        output = self.norm(input)

        return output