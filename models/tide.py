import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from dataclasses import dataclass

from constants import Config
from models.base import ModelBehavior


@dataclass
class TiDEConfig(Config):
    r_hat: int = 4
    decoder_output_dim: int = 4
    hidden_dim: int = 256
    encoder_layer_num: int = 2
    decoder_layer_num: int = 2
    temporal_decoder_hidden: int = 64

# B: Batchsize
# L: Lookback
# H: Horizon
# N: the number of series
# r: the number of covariates for each series
# r_hat: temporalWidth in the paper, i.e., \hat{r} << r
# p: decoderOutputDim in the paper
# hidden_dim: hiddenSize in the paper


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.linear_1 = nn.Linear(in_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, out_dim)
        self.linear_res = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.layernorm = nn.LayerNorm(out_dim)

    def forward(self, x):
        # x: [B,L,in_dim] or [B,in_dim]
        h = F.relu(self.linear_1(x))  # [B,L,in_dim] -> [B,L,hidden_dim] or [B,in_dim] -> [B,hidden_dim]
        h = self.dropout(self.linear_2(h))  # [B,L,hidden_dim] -> [B,L,out_dim] or [B,hidden_dim] -> [B,out_dim]
        res = self.linear_res(x)  # [B,L,in_dim] -> [B,L,out_dim] or [B,in_dim] -> [B,out_dim]
        out = self.layernorm(h+res)  # [B,L,out_dim] or [B,out_dim] 

        # out: [B,L,out_dim] or [B,out_dim]
        return out


class Encoder(nn.Module):
    def __init__(self, layer_num, hidden_dim, r_hat, L, H):
        super(Encoder, self).__init__()
        self.encoder_layer_num = layer_num
        self.horizon = H
        self.first_encoder_layer = ResidualBlock(L + 1 + (L + H) * r_hat, hidden_dim, hidden_dim)
        self.other_encoder_layers = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim, hidden_dim) for _ in range(layer_num-1)
            ])

    def forward(self, x):
        # x: [B*N,L], covariates: [B*N,1], attributes: [B*N,L+H,r]

        # Concat
        e = x

        # Dense Encoder
        e = self.first_encoder_layer(e)  # [B*N,L+1+(L+H)*r_hat] -> [B*N,hidden_dim]
        for i in range(self.encoder_layer_num-1):
            e = self.other_encoder_layers[i](e)  # [B*N,hidden_dim] -> [B*N,hidden_dim]

        # e: [B*N,hidden_dim], covariates_future: [B*N,H,r_hat]
        return e


class Decoder(nn.Module):
    def __init__(self, layer_num, hidden_dim, r_hat, pred_len, decoder_output_dim, temporal_decoder_hidden):
        super(Decoder, self).__init__()
        self.decoder_layer_num = layer_num
        self.horizon = pred_len
        self.last_decoder_layer = ResidualBlock(hidden_dim, hidden_dim, decoder_output_dim * pred_len)
        self.other_decoder_layers = nn.ModuleList([
                ResidualBlock(hidden_dim, hidden_dim, hidden_dim) for _ in range(layer_num-1)
            ])
        self.temporaldecoder = ResidualBlock(decoder_output_dim + r_hat, temporal_decoder_hidden, 1)

    def forward(self, e):
        # e: [B*N,hidden_dim], covariates_future: [B*N,H,r_hat]

        # Dense Decoder
        for i in range(self.decoder_layer_num-1):
            e = self.other_decoder_layers[i](e)  # [B*N,hidden_dim] -> [B*N,hidden_dim]
        g = self.last_decoder_layer(e)  # [B*N,hidden_dim] -> [B*N,p*H]

        # Unflatten
        matrixD = rearrange(g, 'b (h p) -> b h p', h=self.horizon)  # [B*N,p*H] -> [B*N,H,p]

        # Stack
        out = matrixD

        # Temporal Decoder
        out = self.temporaldecoder(out)  # [B*N,H,p+r_hat] -> [B*N,H,1]
        
        # out: [B*N,H,1]
        return out


class TiDE(ModelBehavior):
    def __init__(
            self,
            config: TiDEConfig
        ):
        super(TiDE, self).__init__(config)
        
        encoder_layer_num = config.encoder_layer_num
        decoder_layer_num = config.decoder_layer_num
        hidden_dim = config.hidden_dim
        temporal_decoder_hidden = config.temporal_decoder_hidden
        decoder_output_dim = config.decoder_output_dim
        r_hat = config.r_hat

        seq_len = config.seq_len
        pred_len = config.pred_len

        self.encoder = Encoder(encoder_layer_num, hidden_dim, r_hat, seq_len, pred_len)
        self.decoder = Decoder(decoder_layer_num, hidden_dim, r_hat, pred_len, decoder_output_dim, temporal_decoder_hidden)
        self.residual = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [B,L,N], covariates: [B,L+H,N,r]
        batch_size = x.size(0)
        
        # Channel Independence: Convert Multivariate series to Univariate series
        x = rearrange(x, 'b l n -> (b n) l')  # [B,L,N] -> [B*N,L]
        
        # Encoder
        e = self.encoder(x)

        # Decoder
        out = self.decoder(e)  # out: [B*N,H,1]

        # Global Residual
        prediction = out.squeeze(-1) + self.residual(x)  # prediction: [B*N,H]

        # Reshape
        prediction = rearrange(prediction, '(b n) h -> b h n', b=batch_size)  # [B*N,H] -> [B,H,N]

        # prediction: [B,H,N]
        return prediction