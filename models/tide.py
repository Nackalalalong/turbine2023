import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from dataclasses import dataclass

from constants import Config, N_CHANNEL
from models.base import ModelBehavior


torch.set_default_dtype(torch.float64)

@dataclass(kw_only=True)
class TiDEConfig(Config):
    decoder_output_dim: int
    hidden_dim: int
    encoder_layer_num: int
    decoder_layer_num: int
    temporal_decoder_hidden: int
    dropout_rate: float
    temporal_width: int = 4
    diameter: int = None


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

        return out


class Encoder(nn.Module):
    def __init__(self, layer_num, hidden_dim, seq_len, pred_len, dropout_rate, has_attribute):
        super(Encoder, self).__init__()
        self.encoder_layer_num = layer_num
        self.horizon = pred_len
        
        attr_dim = 0
        if has_attribute:
            attr_dim = 1
        self.first_encoder_layer = ResidualBlock(seq_len + attr_dim, hidden_dim, hidden_dim, dropout_rate)
        self.other_encoder_layers = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim, hidden_dim, dropout_rate) for _ in range(layer_num-1)
            ])

    def forward(self, x, attributes: torch.TensorType = None):
        # x: [B*N,L], covariates: [B*N,1], attributes: [B*N,L+H,r]

        # Concat
        # e = x
        if attributes is not None:
            e = torch.cat([x, attributes], dim=1)  # [B*N,L+1+(L+H)*r_hat]
        else:
            e = x

        # Dense Encoder
        e = self.first_encoder_layer(e)  # [B*N,L] -> [B*N,hidden_dim]
        for i in range(self.encoder_layer_num-1):
            e = self.other_encoder_layers[i](e)  # [B*N,hidden_dim] -> [B*N,hidden_dim]

        # e: [B*N,hidden_dim], covariates_future: [B*N,H,temporal_width]
        return e


class Decoder(nn.Module):
    def __init__(self, layer_num, hidden_dim, pred_len, decoder_output_dim, temporal_decoder_hidden, dropout_rate):
        super(Decoder, self).__init__()
        self.decoder_layer_num = layer_num
        self.horizon = pred_len
        self.last_decoder_layer = ResidualBlock(hidden_dim, hidden_dim, decoder_output_dim * pred_len, dropout_rate)
        self.other_decoder_layers = nn.ModuleList([
                ResidualBlock(hidden_dim, hidden_dim, hidden_dim, dropout_rate) for _ in range(layer_num-1)
            ])
        self.temporaldecoder = ResidualBlock(decoder_output_dim, temporal_decoder_hidden, 1, dropout_rate)

    def forward(self, e):
        # e: [B*N,hidden_dim], covariates_future: [B*N,H,temporal_width]

        # Dense Decoder
        for i in range(self.decoder_layer_num-1):
            e = self.other_decoder_layers[i](e)  # [B*N,hidden_dim] -> [B*N,hidden_dim]
        g = self.last_decoder_layer(e)  # [B*N,hidden_dim] -> [B*N,p*H]

        # Unflatten
        matrixD = rearrange(g, 'b (h p) -> b h p', h=self.horizon)  # [B*N,p*H] -> [B*N,H,p]

        # Stack
        out = matrixD

        # Temporal Decoder
        out = self.temporaldecoder(out)  # [B*N,H,p+temporal_width] -> [B*N,H,1]
        
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
        dropout_rate = config.dropout_rate
        has_attribute = config.diameter is not None

        seq_len = config.seq_len
        pred_len = config.pred_len

        self.encoder = Encoder(encoder_layer_num, hidden_dim, seq_len, pred_len, dropout_rate, has_attribute)
        self.decoder = Decoder(decoder_layer_num, hidden_dim, pred_len, decoder_output_dim, temporal_decoder_hidden, dropout_rate)
        self.residual = nn.Linear(seq_len, pred_len)

    def forward(self, x, attributes: torch.TensorType = None):
        # x: [B,L,N], covariates: [B,L+H,N,r]
        batch_size = x.size(0)
        
        # Channel Independence: Convert Multivariate series to Univariate series
        x = rearrange(x, 'b l n -> (b n) l')  # [B,L,N] -> [B*N,L]

        attributes = None
        if self.config.diameter is not None:
            # attributes = rearrange(attributes, 'b n 1 -> (b n) 1')  # [B,N,1] -> [B*N,1]
            attributes = torch.ones((batch_size * N_CHANNEL, 1), device='cuda') * self.config.diameter
        
        # Encoder
        e = self.encoder(x, attributes)

        # Decoder
        out = self.decoder(e)  # out: [B*N,H,1]

        # Global Residual
        prediction = out.squeeze(-1) + self.residual(x)  # prediction: [B*N,H]

        # Reshape
        prediction = rearrange(prediction, '(b n) h -> b h n', b=batch_size)  # [B*N,H] -> [B,H,N]

        # prediction: [B,H,N]
        return prediction