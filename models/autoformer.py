import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos, DataEmbedding_wo_pos_temp, DataEmbedding_wo_temp
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp


class Autoformer(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """

    def __init__(self, config):
        super(Autoformer, self).__init__()
        self.seq_len = config.seq_len
        self.label_len = config.label_len
        self.pred_len = config.pred_len
        self.output_attention = False

        # Decomp
        kernel_size = config.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        if config.embed_type == 0:
            self.enc_embedding = DataEmbedding_wo_pos(config.enc_in, config.d_model, config.embed,
                                                      config.freq, config.dropout)
            self.dec_embedding = DataEmbedding_wo_pos(config.dec_in, config.d_model, config.embed,
                                                      config.freq, config.dropout)
        elif config.embed_type == 1:
            self.enc_embedding = DataEmbedding(config.enc_in, config.d_model, config.embed,
                                               config.freq, config.dropout)
            self.dec_embedding = DataEmbedding(config.dec_in, config.d_model, config.embed,
                                               config.freq, config.dropout)
        elif config.embed_type == 2:
            self.enc_embedding = DataEmbedding_wo_pos(config.enc_in, config.d_model, config.embed,
                                                      config.freq, config.dropout)
            self.dec_embedding = DataEmbedding_wo_pos(config.dec_in, config.d_model, config.embed,
                                                      config.freq, config.dropout)

        elif config.embed_type == 3:
            self.enc_embedding = DataEmbedding_wo_temp(config.enc_in, config.d_model, config.embed,
                                                       config.freq, config.dropout)
            self.dec_embedding = DataEmbedding_wo_temp(config.dec_in, config.d_model, config.embed,
                                                       config.freq, config.dropout)
        elif config.embed_type == 4:
            self.enc_embedding = DataEmbedding_wo_pos_temp(config.enc_in, config.d_model,
                                                           config.embed, config.freq,
                                                           config.dropout)
            self.dec_embedding = DataEmbedding_wo_pos_temp(config.dec_in, config.d_model,
                                                           config.embed, config.freq,
                                                           config.dropout)

        # Encoder
        self.encoder = Encoder([
            EncoderLayer(AutoCorrelationLayer(
                AutoCorrelation(False,
                                config.factor,
                                attention_dropout=config.dropout,
                                output_attention=self.output_attention), config.d_model,
                config.n_heads),
                         config.d_model,
                         config.d_ff,
                         moving_avg=config.moving_avg,
                         dropout=config.dropout,
                         activation=config.activation) for l in range(config.e_layers)
        ],
                               norm_layer=my_Layernorm(config.d_model))
        # Decoder
        self.decoder = Decoder([
            DecoderLayer(
                AutoCorrelationLayer(
                    AutoCorrelation(True,
                                    config.factor,
                                    attention_dropout=config.dropout,
                                    output_attention=False), config.d_model, config.n_heads),
                AutoCorrelationLayer(
                    AutoCorrelation(False,
                                    config.factor,
                                    attention_dropout=config.dropout,
                                    output_attention=False), config.d_model, config.n_heads),
                config.d_model,
                config.c_out,
                config.d_ff,
                moving_avg=config.moving_avg,
                dropout=config.dropout,
                activation=config.activation,
            ) for l in range(config.d_layers)
        ],
                               norm_layer=my_Layernorm(config.d_model),
                               projection=nn.Linear(config.d_model, config.c_out, bias=True))

    def forward(self,
                x_enc,
                x_mark_enc,
                x_dec,
                x_mark_dec,
                enc_self_mask=None,
                dec_self_mask=None,
                dec_enc_mask=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        #pdb.set_trace()
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out,
                                                 enc_out,
                                                 x_mask=dec_self_mask,
                                                 cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
