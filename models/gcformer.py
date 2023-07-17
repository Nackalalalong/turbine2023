from typing import Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass

from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp
from layers.FourierCorrelation import *
from layers.SelfAttention_Family import AttentionLayer, ProbAttention
from layers.global_conv import Film, FNO, GConv
from layers.RevIN import RevIN
from layers.TCN import TemporalConvNet
from models.autoformer import Autoformer
from models.base import ModelBehavior
from constants import Config


@dataclass(kw_only=True)
class GCFormerConfig(Config):
    max_seq_len: Optional[int] = 1024
    d_k: Optional[int] = None


class GCFormer(ModelBehavior):

    def __init__(self,
                 config: GCFormerConfig,
                 d_v: Optional[int] = None,
                 norm: str = 'BatchNorm',
                 attn_dropout: float = 0.,
                 act: str = "gelu",
                 key_padding_mask: bool = 'auto',
                 padding_var: Optional[int] = None,
                 attn_mask: Optional[Tensor] = None,
                 res_attention: bool = True,
                 pre_norm: bool = False,
                 store_attn: bool = False,
                 pe: str = 'zeros',
                 learn_pe: bool = True,
                 pretrain_head: bool = False,
                 head_type='flatten',
                 verbose: bool = False,
                 **kwargs):

        super().__init__()

        # load parameters
        c_in = config.enc_in
        n_layers = config.e_layers
        n_heads = config.n_heads
        d_model = config.d_model
        d_ff = config.d_ff
        dropout = config.dropout
        fc_dropout = config.fc_dropout
        head_dropout = config.head_dropout
        individual = config.individual
        patch_len = config.patch_len
        stride = config.stride
        padding_patch = config.padding_patch
        revin = config.local_revin
        affine = config.affine
        subtract_last = config.subtract_last
        decomposition = config.decomposition
        kernel_size = config.kernel_size
        target_window = config.pred_len

        self.context_window = config.context_len
        self.batch_size = config.batch_size
        self.enc_in = config.enc_in
        self.context_len = config.context_len
        self.pred_len = config.pred_len
        self.seq_len = config.seq_len
        self.norm_type = config.norm_type
        self.global_model = config.global_model
        self.atten_bias = config.atten_bias
        self.TC_bias = config.TC_bias
        self.h_token = config.h_token
        self.h_channel = config.h_channel
        patch_num = int((self.context_window - patch_len) / stride + 1)
        if padding_patch == 'end':
            patch_num += 1

        max_seq_len = config.max_seq_len
        d_k = config.d_k

        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(c_in=c_in,
                                                 context_window=self.context_window,
                                                 target_window=target_window,
                                                 patch_len=patch_len,
                                                 stride=stride,
                                                 max_seq_len=max_seq_len,
                                                 n_layers=n_layers,
                                                 d_model=d_model,
                                                 n_heads=n_heads,
                                                 d_k=d_k,
                                                 d_v=d_v,
                                                 d_ff=d_ff,
                                                 norm=norm,
                                                 attn_dropout=attn_dropout,
                                                 dropout=dropout,
                                                 act=act,
                                                 key_padding_mask=key_padding_mask,
                                                 padding_var=padding_var,
                                                 attn_mask=attn_mask,
                                                 res_attention=res_attention,
                                                 pre_norm=pre_norm,
                                                 store_attn=store_attn,
                                                 pe=pe,
                                                 learn_pe=learn_pe,
                                                 fc_dropout=fc_dropout,
                                                 head_dropout=head_dropout,
                                                 padding_patch=padding_patch,
                                                 pretrain_head=pretrain_head,
                                                 head_type=head_type,
                                                 individual=individual,
                                                 revin=revin,
                                                 affine=affine,
                                                 subtract_last=subtract_last,
                                                 verbose=verbose,
                                                 **kwargs)
            self.model_res = PatchTST_backbone(c_in=c_in,
                                               context_window=self.context_window,
                                               target_window=target_window,
                                               patch_len=patch_len,
                                               stride=stride,
                                               max_seq_len=max_seq_len,
                                               n_layers=n_layers,
                                               d_model=d_model,
                                               n_heads=n_heads,
                                               d_k=d_k,
                                               d_v=d_v,
                                               d_ff=d_ff,
                                               norm=norm,
                                               attn_dropout=attn_dropout,
                                               dropout=dropout,
                                               act=act,
                                               key_padding_mask=key_padding_mask,
                                               padding_var=padding_var,
                                               attn_mask=attn_mask,
                                               res_attention=res_attention,
                                               pre_norm=pre_norm,
                                               store_attn=store_attn,
                                               pe=pe,
                                               learn_pe=learn_pe,
                                               fc_dropout=fc_dropout,
                                               head_dropout=head_dropout,
                                               padding_patch=padding_patch,
                                               pretrain_head=pretrain_head,
                                               head_type=head_type,
                                               individual=individual,
                                               revin=revin,
                                               affine=affine,
                                               subtract_last=subtract_last,
                                               verbose=verbose,
                                               **kwargs)
        else:
            self.model = PatchTST_backbone(c_in=c_in,
                                           context_window=self.context_window,
                                           target_window=target_window,
                                           patch_len=patch_len,
                                           stride=stride,
                                           max_seq_len=max_seq_len,
                                           n_layers=n_layers,
                                           d_model=d_model,
                                           n_heads=n_heads,
                                           d_k=d_k,
                                           d_v=d_v,
                                           d_ff=d_ff,
                                           norm=norm,
                                           attn_dropout=attn_dropout,
                                           dropout=dropout,
                                           act=act,
                                           key_padding_mask=key_padding_mask,
                                           padding_var=padding_var,
                                           attn_mask=attn_mask,
                                           res_attention=res_attention,
                                           pre_norm=pre_norm,
                                           store_attn=store_attn,
                                           pe=pe,
                                           learn_pe=learn_pe,
                                           fc_dropout=fc_dropout,
                                           head_dropout=head_dropout,
                                           padding_patch=padding_patch,
                                           pretrain_head=pretrain_head,
                                           head_type=head_type,
                                           individual=individual,
                                           revin=revin,
                                           affine=affine,
                                           subtract_last=subtract_last,
                                           verbose=verbose,
                                           **kwargs)

        self.linear_seq_pred = nn.Linear(config.seq_len, config.pred_len, bias=True)
        self.linear_channel_out = nn.Linear(self.h_channel, config.enc_in, bias=True)
        self.linear_channel_in = nn.Linear(config.enc_in, self.h_channel, bias=True)
        self.linear_token_in = nn.Linear(config.pred_len, self.h_token, bias=True)
        self.linear_token_out = nn.Linear(self.h_token, config.pred_len, bias=True)
        self.linear_local_token = nn.Linear(config.context_len, config.pred_len, bias=True)
        self.norm_channel = nn.BatchNorm1d(self.h_channel)
        self.norm_token = nn.BatchNorm1d(self.h_token)
        self.ff = nn.Sequential(nn.GELU(), nn.Dropout(config.fc_dropout))

        decoder_cross_att = ProbAttention()
        self.decoder_channel = AttentionLayer(decoder_cross_att, self.h_channel, config.n_heads)
        self.decoder_token = AttentionLayer(decoder_cross_att, self.h_token, config.n_heads)

        if config.global_model == 'Gconv':
            self.global_layer_Gconv = GConv(config.batch_size,
                                            d_model=config.enc_in,
                                            d_state=config.enc_in,
                                            l_max=config.seq_len,
                                            channels=config.n_heads,
                                            bidirectional=True,
                                            kernel_dim=32,
                                            n_scales=None,
                                            decay_min=2,
                                            decay_max=2,
                                            transposed=False)
        elif config.global_model == 'FNO':
            self.global_layer_FNO = nn.Sequential(FNO(1, 1, self.enc_in), nn.GELU(),
                                                  nn.Dropout(config.dropout))
        elif config.global_model == 'Film':
            self.global_layer_Film = nn.Sequential(
                Film(1, 1, self.enc_in, self.seq_len, self.pred_len), nn.GELU(),
                nn.Dropout(config.dropout))

        self.revin_layer = RevIN(config.enc_in, affine=True, subtract_last=False)
        self.TCN = TemporalConvNet(config.enc_in, [config.h_channel, config.enc_in])
        self.local_Autoformer = Autoformer.Model(config)
        self.local_bias = nn.Parameter(torch.rand(1) * 0.1 + config.local_bias)
        self.global_bias = nn.Parameter(torch.rand(1) * 0.1 + config.global_bias)

    def forward(self, x):  # x: [Batch, Input length, Channel]

        ################### norm
        seq_last = x[:, -1:, :].detach()
        if self.norm_type == 'revin':
            x = self.revin_layer(x, 'norm')
        elif self.norm_type == 'seq_last':
            x = x - seq_last
        global_x = x
        local_x = x[:, -self.context_len:, :]

        ################### Encoder: global branch
        if self.global_model == 'Gconv':
            global_x = self.global_layer_Gconv(global_x, return_kernel=False)
            global_x = self.linear_seq_pred(global_x.permute(0, 2, 1)).permute(0, 2, 1)
        elif self.global_model == 'FNO':
            global_x = global_x.permute(0, 2, 1).unsqueeze(3)
            global_x = self.global_layer_FNO(global_x)
            global_x = self.linear_seq_pred(global_x.squeeze(2)).permute(0, 2, 1)
        elif self.global_model == 'Film':
            global_x = global_x.permute(0, 2, 1).unsqueeze(3)
            global_x = self.global_layer_Film(global_x)
            global_x = self.linear_seq_pred(global_x.squeeze(2)).permute(0, 2, 1)

        ################### Encoder: local branch
        if self.decomposition:
            res_init, trend_init = self.decomp_module(local_x)
            res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            local_x = res + trend
            local_x = local_x.permute(0, 2, 1)  # x: [Batch, pred length, Channel]
        else:
            local_x = local_x.permute(0, 2, 1)
            local_x = self.model(local_x)
            local_x = local_x.permute(0, 2, 1)

        ################### Decoder
        global_x_channel = self.linear_channel_in(global_x)
        local_x_channel = self.linear_channel_in(local_x)
        output_channel_l = self.ff(
            self.decoder_channel(global_x_channel, local_x_channel,
                                 local_x_channel)) + local_x_channel
        output_channel_g = self.ff(
            self.decoder_channel(local_x_channel, global_x_channel,
                                 global_x_channel)) + global_x_channel
        output_channel_l = self.norm_channel(output_channel_l.permute(0, 2, 1)).permute(0, 2, 1)
        output_channel_g = self.norm_channel(output_channel_g.permute(0, 2, 1)).permute(0, 2, 1)
        output_channel = self.atten_bias * output_channel_l + (1 -
                                                               self.atten_bias) * output_channel_g
        output_channel = self.ff(self.linear_channel_out(output_channel))

        global_x_token = self.linear_token_in(global_x.permute(0, 2, 1))
        local_x_token = self.linear_token_in(local_x.permute(0, 2, 1))
        output_token_l = self.ff(self.decoder_token(global_x_token, local_x_token,
                                                    local_x_token)) + local_x_token
        output_token_g = self.ff(self.decoder_token(local_x_token, global_x_token,
                                                    global_x_token)) + global_x_token
        output_token_l = self.norm_token(output_token_l.permute(0, 2, 1)).permute(0, 2, 1)
        output_token_g = self.norm_token(output_token_g.permute(0, 2, 1)).permute(0, 2, 1)
        output_token = self.atten_bias * output_token_l + (1 - self.atten_bias) * output_token_g
        output_token = self.ff(self.linear_token_out(output_token).permute(0, 2, 1))

        output = self.TC_bias * output_channel + (
            1 -
            self.TC_bias) * output_token + self.global_bias * global_x + self.local_bias * local_x

        ################### denorm
        if self.norm_type == 'revin':
            output = self.revin_layer(output, 'denorm')
        elif self.norm_type == 'seq_last':
            output = output + seq_last
        return output, local_x, global_x, self.global_bias, self.local_bias
