import torch
import torch.nn.functional as F

from torch import nn

EPS = 1e-12


class GALR(nn.Module):
    def __init__(self,
                 num_features,
                 hidden_channels,
                #num_blocks=3, #size change
                 num_heads=6,
                 norm=True,
                 dropout=1e-1,
                 low_dimension=True,
                 causal=True,
                 eps=EPS,
                 **kwargs):
        super().__init__()

        # Network configuration
        net = []

        num_blocks = kwargs['num_blocks']
        for _ in range(num_blocks-1):
            net.append(GALRBlock(num_features,
                                 hidden_channels,
                                 num_heads=num_heads,
                                 norm=norm,
                                 dropout=dropout,
                                 low_dimension=low_dimension,
                                 causal=causal,
                                 eps=eps,
                                 **kwargs
                                 )
                       )

        net.append(GALRBlock(num_features,
                             hidden_channels,
                             num_heads=num_heads,
                             norm=norm,
                             dropout=dropout,
                             low_dimension=low_dimension,
                             causal=causal,
                             eps=eps,
                             attn_op=True,
                             **kwargs)
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


class GALRBlock(nn.Module):
    def __init__(self,
                 num_features,
                 hidden_channels,
                 num_heads=8,
                 causal=True,
                 norm=True,
                 dropout=1e-1,
                 low_dimension=True,
                 eps=EPS,
                 attn_op=False,
                 **kwargs):
        super().__init__()
        self.attn_op = attn_op

        self.intra_chunk_block = IntraChunkRNN(num_features, hidden_channels=hidden_channels, norm=norm)

        if low_dimension:
            chunk_size = kwargs['chunk_size']
            down_chunk_size = kwargs['down_chunk_size']
            self.inter_chunk_block = LowDimensionGloballyAttentiveBlock(num_features,
                                                                        chunk_size=chunk_size,
                                                                        down_chunk_size=down_chunk_size,
                                                                        num_heads=num_heads,
                                                                        causal=causal,
                                                                        norm=norm,
                                                                        dropout=dropout,
                                                                        attn_op=attn_op,
                                                                        eps=eps)
        else:
            self.inter_chunk_block = GloballyAttentiveBlock(num_features,
                                                            num_heads=num_heads,
                                                            causal=causal,
                                                            norm=norm,
                                                            dropout=dropout,
                                                            attn_op=attn_op,
                                                            eps=eps)

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


class GloballyAttentiveBlock(GloballyAttentiveBlockBase):
    def __init__(self, num_features, num_heads=8, causal=True, norm=True, dropout=1e-1, eps=EPS, attn_op=False):
        super().__init__()

        self.norm = norm
        self.attn_op = attn_op

        if self.norm:
            self.norm2d_in = LayerNormAlongChannel(num_features, eps=eps)

        self.multihead_attn = nn.MultiheadAttention(num_features, num_heads)
        #self.multihead_attn = MultiheadAttentionModule(num_features,num_heads)##for new features

        if dropout is not None:
            self.dropout = True
            self.dropout1d = nn.Dropout(p=dropout)
        else:
            self.dropout = False

        if self.norm:
            norm_layer = CumulativeLayerNorm1d if causal else GlobalLayerNorm
            self.norm2d_out = norm_layer(num_features, eps=eps)

    def forward(self, input_x):
        """
        Args:
            input_x (batch_size, num_features, S, K): K is chunk size
        Returns:
            output (batch_size, num_features, S, K)
        """
        batch_size, num_features, S, K = input_x.size()

        if self.norm:
            x = self.norm2d_in(input_x)  # -> (batch_size, num_features, S, K)
        else:
            x = input_x
        encoding = self.positional_encoding(length=S * K, dimension=num_features).permute(1, 0).view(num_features, S, K).to(x.device)
        x = x + encoding  # -> (batch_size, num_features, S, K)
        x = x.permute(2, 0, 3, 1).contiguous()  # -> (S, batch_size, K, num_features)
        x = x.view(S, batch_size * K, num_features)  # -> (S, batch_size*K, num_features)

        residual = x  # (S, batch_size*K, num_features)
        x, attention = self.multihead_attn(x, x, x)  # (T_tgt, batch_size, num_features), (batch_size, T_tgt, T_src), where T_tgt = T_src = T
        ##should use this attention
        if self.dropout:
            x = self.dropout1d(x)
        x = x + residual  # -> (S, batch_size*K, num_features)
        x = x.view(S, batch_size, K, num_features)
        x = x.permute(1, 3, 0, 2).contiguous()  # -> (batch_size, num_features, S, K)

        if self.norm:
            x = self.norm2d_out(x)  # -> (batch_size, num_features, S, K)
        x = x + input_x
        output = x.view(batch_size, num_features, S, K)

        if not self.attn_op:
            return output
        else:
            return output, attention


class LowDimensionGloballyAttentiveBlock(GloballyAttentiveBlockBase):
    def __init__(self, num_features, chunk_size=100, down_chunk_size=100, num_heads=8, causal=True, norm=True,
                 dropout=1e-1, attn_op=False, eps=EPS):
        super().__init__()

        self.down_chunk_size = down_chunk_size
        self.norm = norm
        self.attn_op = attn_op

        self.fc_map = nn.Linear(chunk_size, down_chunk_size)

        if self.norm:
            self.norm2d_in = LayerNormAlongChannel(num_features, eps=eps)

        self.multihead_attn = nn.MultiheadAttention(num_features, num_heads)

        if dropout is not None:
            self.dropout = True
            self.dropout1d = nn.Dropout(p=dropout)
        else:
            self.dropout = False

        if self.norm:
            norm_layer = CumulativeLayerNorm1d if causal else GlobalLayerNorm
            self.norm2d_out = norm_layer(num_features, eps=eps)

        self.fc_inv = nn.Linear(down_chunk_size, chunk_size)

    def forward(self, input_x):
        """
        Args:
            input (batch_size, num_features, S, K): K is chunk size
        Returns:
            output (batch_size, num_features, S, K)
        """
        Q = self.down_chunk_size
        batch_size, num_features, S, K = input_x.size()

        x = self.fc_map(input_x)  # (batch_size, num_features, S, K) -> (batch_size, num_features, S, Q)

        if self.norm:
            x = self.norm2d_in(x)  # -> (batch_size, num_features, S, Q)

        encoding = self.positional_encoding(length=S * Q, dimension=num_features).permute(1, 0).view(num_features, S, Q).to(x.device)
        x = x + encoding  # -> (batch_size, num_features, S, Q)
        x = x.permute(2, 0, 3, 1).contiguous()  # -> (S, batch_size, Q, num_features)
        x = x.view(S, batch_size * Q, num_features)  # -> (S, batch_size*Q, num_features)

        residual = x  # (S, batch_size*Q, num_features)
        x, attention = self.multihead_attn(x, x, x)  # (T_tgt, batch_size, num_features), (batch_size, T_tgt, T_src), where T_tgt = T_src = T

        if self.dropout:
            x = self.dropout1d(x)
        x = x + residual  # -> (S, batch_size*Q, num_features)
        x = x.view(S, batch_size, Q, num_features)
        x = x.permute(1, 3, 0, 2).contiguous()  # -> (batch_size, num_features, S, Q)

        if self.norm:
            x = self.norm2d_out(x)  # -> (batch_size, num_features, S, Q)

        x = self.fc_inv(x)  # (batch_size, num_features, S, Q) -> (batch_size, num_features, S, K)
        x = x + input_x
        output = x.view(batch_size, num_features, S, K)

        if self.attn_op:
            return output, attention
        else:
            return output


class LayerNormAlongChannel(nn.Module):
    def __init__(self, num_features, eps=EPS):
        super().__init__()

        self.num_features = num_features
        self.eps = eps

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

    def __repr__(self):
        s = '{}'.format(self.__class__.__name__)
        s += '({num_features}, eps={eps})'

        return s.format(**self.__dict__)


class IntraChunkRNN(nn.Module):
    def __init__(self, num_features, hidden_channels, norm=True, rnn_type='lstm', bidirectional=False):
        super().__init__()

        self.num_features, self.hidden_channels = num_features, hidden_channels
        if bidirectional:
            num_directions = 2  # bi-direction
        else:
            num_directions = 1
        self.norm = norm

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=num_features,
                               hidden_size=hidden_channels,
                               batch_first=True,
                               bidirectional=bidirectional)
        else:
            raise NotImplementedError("Not support {}.".format(rnn_type))

        self.fc = nn.Linear(num_directions * hidden_channels, num_features)

        if self.norm:
            self.norm1d = GlobalLayerNorm(num_features)

    def forward(self, input_x):
        """
        Args:
            input_x (batch_size, num_features, S, chunk_size)
        Returns:
            output (batch_size, num_features, S, chunk_size)
        """
        num_features = self.num_features
        batch_size, _, S, chunk_size = input_x.size()

        self.rnn.flatten_parameters()

        residual = input_x  # (batch_size, num_features, S, chunk_size)
        x = input_x.permute(0, 2, 3, 1).contiguous()  # -> (batch_size, S, chunk_size, num_features)
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


class GlobalLayerNorm(nn.Module):
    def __init__(self, num_features, eps=EPS):
        super().__init__()

        self.num_features = num_features
        self.eps = eps

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

    def __repr__(self):
        s = '{}'.format(self.__class__.__name__)
        s += '({num_features}, eps={eps})'

        return s.format(**self.__dict__)


class CumulativeLayerNorm1d(nn.Module):
    def __init__(self, num_features, eps=EPS):
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
        eps = self.eps

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

        if torch.cuda.is_available():
            cum_num = torch.arange(C, C * (T + 1), C, dtype=torch.float).cuda()  # (T, ): [C, 2*C, ..., T*C]
        else:
            cum_num = torch.arange(C, C * (T + 1), C, dtype=torch.float)  # (T, ): [C, 2*C, ..., T*C]

        cum_mean = cum_sum / cum_num  # (batch_size, T)
        cum_squared_mean = cum_squared_sum / cum_num
        cum_var = cum_squared_mean - cum_mean ** 2

        cum_mean, cum_var = cum_mean.unsqueeze(dim=1), cum_var.unsqueeze(dim=1)

        output = (input_x - cum_mean) / (torch.sqrt(cum_var) + eps) * self.gamma + self.beta

        if n_dims == 4:
            output = output.view(batch_size, C, S, chunk_size)

        return output

    def __repr__(self):
        s = '{}'.format(self.__class__.__name__)
        s += '({num_features}, eps={eps})'

        return s.format(**self.__dict__)


class Segment1d(nn.Module):
    """
    Segmentation. Input tensor is 3-D (audio-like), but output tensor is 4-D (image-like).
    """
    def __init__(self, chunk_size, hop_size):
        super().__init__()

        self.chunk_size, self.hop_size = chunk_size, hop_size

    def forward(self, input_x):
        """
        Args:
            input_x (batch_size, num_features, n_frames)
        Returns:
            output (batch_size, num_features, S, chunk_size): S is length of global output, where S = (n_frames-chunk_size)//hop_size + 1
        """
        chunk_size, hop_size = self.chunk_size, self.hop_size
        batch_size, num_features, n_frames = input_x.size()

        input_x = input_x.view(batch_size, num_features, n_frames, 1)
        x = F.unfold(input_x, kernel_size=(chunk_size, 1), stride=(hop_size, 1)) # -> (batch_size, num_features*chunk_size, S), where S = (n_frames-chunk_size)//hop_size+1

        x = x.view(batch_size, num_features, chunk_size, -1)
        output = x.permute(0, 1, 3, 2).contiguous() # -> (batch_size, num_features, S, chunk_size)

        return output

    def extra_repr(self):
        s = "chunk_size={chunk_size}, hop_size={hop_size}".format(chunk_size=self.chunk_size, hop_size=self.hop_size)
        return s


class MultiheadAttentionModule(nn.Module):
    def __init__(self, input_size, num_heads):
        super().__init__()

        # Multihead Attention의 입력 차원을 맞추기 위해 Linear Layer 추가
        self.linear = nn.Linear(input_size, 8)

        # Multihead Attention 모듈 선언
        self.multihead_attn = nn.MultiheadAttention(8, num_heads)

    def forward(self, x):
        # 입력 텐서 x의 크기를 가져옴
        seq_len, batch_size, input_size = x.size()

        # 입력 텐서를 Linear Layer를 통과시켜 차원을 맞춤
        x = self.linear(x.view(seq_len * batch_size, input_size))
        x = x.view(seq_len, batch_size, 8)

        # Multihead Attention 모듈 적용
        attn_output, _ = self.multihead_attn(x, x, x)

        return attn_output
