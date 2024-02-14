import torch
from torch import nn
from torch.nn import functional as F

class Segment1d(nn.Module):
    """Segmentation. Input tensor is 3-D (audio-like), but output tensor is 4-D (image-like)."""
    def __init__(self, chunk_size: int, hop_size: int):
        super().__init__()
        self.chunk_size, self.hop_size = chunk_size, hop_size

    def forward(self, input_x):
        """
        Args:
            input_x (batch_size, num_features, n_frames)
        Returns:
            output (batch_size, num_features, S, chunk_size): S is length of global output, where S = (n_frames-chunk_size)//hop_size + 1
        """
        batch_size, num_features, n_frames = input_x.size()
        input_x = input_x.view(batch_size, num_features, n_frames, 1)
        x = F.unfold(input_x, kernel_size=(self.chunk_size, 1), stride=(self.hop_size, 1)) # -> (batch_size, num_features*chunk_size, S), where S = (n_frames-chunk_size)//hop_size+1
        x = x.view(batch_size, num_features, self.chunk_size, -1)
        output = x.permute(0, 1, 3, 2).contiguous() # -> (batch_size, num_features, S, chunk_size)

        return output

class GALR(nn.Module):
    def __init__(self, num_blocks: int,
                 num_features: int, hidden_channels: int, batch_first: bool, bidirectional: bool,
                 num_heads: int, eps: float, dropout: float):
        super().__init__()

        # Network configuration
        net = []
        for _ in range(num_blocks - 1):
            net.append(GALRBlock(num_features=num_features, hidden_channels=hidden_channels, batch_first=batch_first, bidirectional=bidirectional,
                                 num_heads=num_heads, eps=eps, dropout=dropout, attn_op=False)
                       )
        net.append(GALRBlock(num_features=num_features, hidden_channels=hidden_channels, batch_first=batch_first, bidirectional=bidirectional,
                             num_heads=num_heads, eps=eps, dropout=dropout, attn_op=True)
                   )
        self.net = nn.Sequential(*net)

    def forward(self, input_x):
        """
        Args:
            input_x (batch_size, num_features, S, chunk_size)
        Returns:
            output (batch_size, num_features, S, chunk_size)
        """
        output = self.net(input_x)

        return output

# class GALR -> class GALRBlock
class GALRBlock(nn.Module):
    def __init__(self, num_features: int, hidden_channels: int, batch_first: bool, bidirectional: bool,
                 num_heads: int, eps: float, dropout: float, attn_op: bool):
        super().__init__()
        self.intra_chunk_block = IntraChunkRNN(num_features=num_features, hidden_channels=hidden_channels, batch_first=batch_first, bidirectional=bidirectional, eps=eps)
        self.inter_chunk_block = GloballyAttentiveBlock(num_features=num_features, num_heads=num_heads,
                                                        eps=eps, dropout=dropout, attn_op=attn_op)

    def forward(self, input_x):
        """
        Args:
            input_x (batch_size, num_features, S, chunk_size)
        Returns:
            output (batch_size, num_features, S, chunk_size)
        """
        x = self.intra_chunk_block(input_x)
        output = self.inter_chunk_block(x)

        return output

# GALR -> GALRBlock -> IntraChunkRNN
class IntraChunkRNN(nn.Module):
    def __init__(self, num_features: int, hidden_channels: int, batch_first: bool, bidirectional: bool, eps: float):
        super().__init__()
        self.num_features, self.hidden_channels = num_features, hidden_channels

        if bidirectional:
            num_directions = 2  # bi-direction
        else:
            num_directions = 1

        self.rnn = nn.LSTM(input_size=num_features, hidden_size=hidden_channels, batch_first=batch_first, bidirectional=bidirectional)
        self.fc = nn.Linear(num_directions * hidden_channels, num_features)
        self.norm1d = GlobalLayerNorm(num_features=num_features, eps=eps)

    def forward(self, input_x):
        """
        Args:
            input_x (batch_size, num_features, S, chunk_size)
        Returns:
            output (batch_size, num_features, S, chunk_size)
        """
        batch_size, _, S, chunk_size = input_x.size()
        self.rnn.flatten_parameters()
        residual = input_x  # (batch_size, num_features, S, chunk_size)

        x = input_x.permute(0, 2, 3, 1).contiguous()  # -> (batch_size, S, chunk_size, num_features)
        x = x.view(batch_size * S, chunk_size, self.num_features)
        x, _ = self.rnn(x)  # (batch_size*S, chunk_size, num_features) -> (batch_size*S, chunk_size, num_directions*hidden_channels)
        x = self.fc(x)  # -> (batch_size*S, chunk_size, num_features)
        x = x.view(batch_size, S * chunk_size, self.num_features)  # (batch_size, S*chunk_size, num_features)
        x = x.permute(0, 2, 1).contiguous()  # -> (batch_size, num_features, S*chunk_size)
        x = self.norm1d(x)  # (batch_size, num_features, S*chunk_size)
        x = x.view(batch_size, self.num_features, S, chunk_size)  # -> (batch_size, num_features, S, chunk_size)

        output = x + residual

        return output

# GALR -> GALRBlock -> IntraChunkRNN -> GlobalLayerNorm
class GlobalLayerNorm(nn.Module):
    def __init__(self, num_features: int, eps: float):
        super().__init__()
        self.norm = nn.GroupNorm(1, num_features, eps=eps)

    def forward(self, input_x):
        """
        Args:
            input_x (batch_size, C, *)
        Returns:
            output (batch_size, C, *)
        """
        output = self.norm(input_x)

        return output

# GALR -> class GALRBlock -> (GloballyAttentiveBlockBase) -> GloballyAttentiveBlock
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

        position = torch.arange(length)  # (length,)
        position = position.unsqueeze(dim=1)  # (length, 1)
        index = torch.arange(dimension // 2) / dimension  # (dimension // 2,)
        index = index.unsqueeze(dim=0)  # (1, dimension // 2)
        indices = position / base ** index
        output = torch.cat([torch.sin(indices), torch.cos(indices)], dim=1)

        return output

# GALR -> class GALRBlock -> (GloballyAttentiveBlockBase) -> GloballyAttentiveBlock
class GloballyAttentiveBlock(GloballyAttentiveBlockBase):
    def __init__(self, num_features: int, num_heads: int, eps: float, dropout: float, attn_op: bool):
        super().__init__()
        self.norm2d_in = LayerNormAlongChannel(num_features=num_features, eps=eps)
        self.multihead_attn = nn.MultiheadAttention(num_features, num_heads)
        self.dropout1d = nn.Dropout(p=dropout)
        self.norm2d_out = CumulativeLayerNorm1d(num_features, eps=eps)
        self.attn_op = attn_op

    def forward(self, input_x):
        """
        Args:
            input_x (batch_size, num_features, S, K): K is chunk size
        Returns:
            output (batch_size, num_features, S, K)
        """
        batch_size, num_features, S, K = input_x.size()

        x = self.norm2d_in(input_x)  # -> (batch_size, num_features, S, K)

        encoding = self.positional_encoding(length=S * K, dimension=num_features).permute(1, 0).view(num_features, S, K).to(x.device)
        x = x + encoding  # -> (batch_size, num_features, S, K)
        x = x.permute(2, 0, 3, 1).contiguous()  # -> (S, batch_size, K, num_features)
        x = x.view(S, batch_size * K, num_features)  # -> (S, batch_size*K, num_features)
        residual = x  # (S, batch_size*K, num_features)
        x, attention = self.multihead_attn(x, x, x)  # (T_tgt, batch_size, num_features), (batch_size, T_tgt, T_src), where T_tgt = T_src = T

        # should use this attention
        x = self.dropout1d(x)
        x = x + residual  # -> (S, batch_size*K, num_features)
        x = x.view(S, batch_size, K, num_features)
        x = x.permute(1, 3, 0, 2).contiguous()  # -> (batch_size, num_features, S, K)
        x = self.norm2d_out(x)  # -> (batch_size, num_features, S, K)
        x = x + input_x
        output = x.view(batch_size, num_features, S, K)

        if self.attn_op:
            return output, attention
        else:
            return output

# GALR -> class GALRBlock -> (GloballyAttentiveBlockBase) -> GloballyAttentiveBlock -> LayerNormAlongChannel
class LayerNormAlongChannel(nn.Module):
    def __init__(self, num_features: int, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(num_features, eps=eps)

    def forward(self, input_x):
        """
        Args:
            input_x (batch_size, num_features, *)
        Returns:
            output (batch_size, num_features, *)
        """
        n_dims = input_x.dim()
        dims = list(range(n_dims))
        permuted_dims = dims[0:1] + dims[2:] + dims[1:2]
        x = input_x.permute(*permuted_dims)
        x = self.norm(x)
        permuted_dims = dims[0:1] + dims[-1:] + dims[1:-1]
        output = x.permute(*permuted_dims).contiguous()

        return output

# GALR -> class GALRBlock -> (GloballyAttentiveBlockBase) -> GloballyAttentiveBlock ->  CumulativeLayerNorm1d
class CumulativeLayerNorm1d(nn.Module):
    def __init__(self, num_features: int, eps: float):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.gamma = nn.Parameter(torch.Tensor(1, num_features, 1))
        self.beta = nn.Parameter(torch.Tensor(1, num_features, 1))
        self._reset_parameters()

    def _reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.fill_(0)

    def forward(self, input_x):
        """
        Args:
            input_x (batch_size, C, T) or (batch_size, C, S, chunk_size):
        Returns:
            output (batch_size, C, T) or (batch_size, C, S, chunk_size): same shape as the input
        """
        n_dims = input_x.dim()
        chunk_size = None
        S = None

        if n_dims == 3:
            batch_size, C, T = input_x.size()
        elif n_dims == 4:
            batch_size, C, S, chunk_size = input_x.size()
            T = S * chunk_size
            input_x = input_x.view(batch_size, C, T)
        else:
            raise ValueError("Only support 3D or 4D input, but given {}D".format(input_x.dim()))

        step_sum = torch.sum(input_x, dim=1)  # (batch_size, T)
        step_squared_sum = torch.sum(input_x ** 2, dim=1)  # (batch_size, T)
        cum_sum = torch.cumsum(step_sum, dim=1)  # (batch_size, T)
        cum_squared_sum = torch.cumsum(step_squared_sum, dim=1)  # (batch_size, T)

        '''if torch.cuda.is_available():
            cum_num = torch.arange(C, C * (T + 1), C, dtype=torch.float).cuda()  # (T, ): [C, 2*C, ..., T*C]
        else:
            cum_num = torch.arange(C, C * (T + 1), C, dtype=torch.float)  # (T, ): [C, 2*C, ..., T*C]'''
        cum_num = torch.arange(C, C * (T + 1), C, dtype=torch.float).to(input_x.device)

        cum_mean = cum_sum / cum_num  # (batch_size, T)
        cum_squared_mean = cum_squared_sum / cum_num
        cum_var = cum_squared_mean - cum_mean ** 2
        cum_mean, cum_var = cum_mean.unsqueeze(dim=1), cum_var.unsqueeze(dim=1)
        output = (input_x - cum_mean) / (torch.sqrt(cum_var) + self.eps) * self.gamma + self.beta

        if n_dims == 4:
            output = output.view(batch_size, C, S, chunk_size)

        return output