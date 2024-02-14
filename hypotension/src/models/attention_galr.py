from src.models.cnn import Embedding
from src.models.galr import GALR, Segment1d
from torch import nn
from torch.nn import functional as F
import torch


class XGALR(nn.Module):
    def __init__(self,
                 input_size: int,
                 embedding_dim: int,
                 gembedding_dim: int,
                 sequences: int,
                 num_heads: int,
                 chunk_size: int,
                 hop_size: int,
                 hidden_channels: int,
                 low_dimension: bool,
                 num_classes: int,
                 linear: bool,
                 save_attn: bool,
                 **kwargs):
        super(XGALR, self).__init__()

        self.name = self.__class__.__name__
        self.save_attn = save_attn
        if kwargs['T']:
            self.T = kwargs['T']
        else:
            self.T = 0.5

        # INFO: Multi-head attention layer.
        self.embedding = Embedding(input_size=input_size,
                                   embedding_dim=embedding_dim,
                                   sequences=sequences,
                                   linear=linear,
                                   axis=1)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)

        # INFO: GALR layers
        self.gembedding = Embedding(input_size=input_size,
                                    embedding_dim=gembedding_dim,
                                    sequences=sequences,
                                    linear=linear)

        self.galr = nn.Sequential(
            Segment1d(chunk_size=chunk_size, hop_size=hop_size),
            GALR(num_features=input_size,
                 hidden_channels=hidden_channels,
                 low_dimension=low_dimension,
                 num_heads=num_heads,
                 **kwargs)
        )

        self.bn = nn.BatchNorm1d(input_size)
        #self.pre = nn.Linear(in_features=7, out_features=8, bias=False)

        # TRY: Change this layer to FC
        self.conv1d = nn.Conv1d(input_size, 1, kernel_size=1, stride=1)

        # TODO: Implement suitable FC layer in the future.
        self.fc = nn.Sequential(
            nn.Linear(sequences, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_x):
        #input_x = self.pre(input_x)

        batch, sequences, features = input_x.size()

        zero_padding = torch.zeros(batch,sequences, 1)
        zero_padding = zero_padding.to('cuda')

        input_x = torch.cat([input_x,zero_padding], dim=2)
        features = 8


        # INFO: Multi-head attention embedding inputs
        embedded_out = self.embedding(input_x)      # embedded output for multihead_attention
        embedded_out = self.bn(embedded_out)
        mthd_out, attention_weights = self.multihead_attention(embedded_out, embedded_out, embedded_out)

        # INFO: GALR embedding inputs
        embedded_out = self.gembedding(input_x)     #embedded output for galr block
        galr_out, gatt = self.galr(embedded_out)
        galr_out = galr_out.view(batch, features, -1)

        # INFO Temperature value
        out = torch.bmm(self.T*attention_weights, galr_out)

        # out = out + galr_out
        # out = F.tanh(self.conv1d(out))
        out = F.relu(self.conv1d(out))
        out = out.view(batch, -1)
        out = self.fc(out)

        if self.save_attn:
            out = (out, attention_weights, gatt)

        return out
