import torch
import torch.nn as nn
from layers.Transformer_EncDec import MFRSEncoder, MFRSEncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import MFRSEmbedding
import torch.nn.functional as F


class Model(nn.Module):
    """
    Paper link: http://arxiv.org/abs/2503.08328
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.use_dc = configs.use_dc
        
        # Embedding
        self.enc_embedding = MFRSEmbedding(configs.use_dc, configs.use_embed, configs.enc_in, configs.seq_len, configs.rs_len, configs.d_model, 
                                                    configs.dropout)

        # self.dc_in = dc_encoding(configs.seq_len, configs.enc_in)
        # self.dc_out = dc_encoding(configs.pred_len, configs.enc_in)
        # self.dc_projection = nn.Linear(configs.seq_len, configs.seq_len)
        
        # Encoder-only architecture
        self.encoder = MFRSEncoder(
            [
                MFRSEncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_refer):

        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
            
            means_xr = x_refer.mean(1, keepdim=True).detach()
            x_refer = x_refer - means_xr
            stdev_xr = torch.sqrt(torch.var(x_refer, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_refer /= stdev_xr                                     

        # if self.use_dc:
        #     x_enc = x_enc - self.dc_projection(x_enc.permute(0, 2, 1)).permute(0, 2, 1)
        
        x, x_refer = self.enc_embedding(x_enc, x_refer)        
        enc_out, attns = self.encoder(x, x_refer, attn_mask=None)       
        dec_out = self.projector(enc_out)
        dec_out = dec_out.permute(0, 2, 1)           
                
        # if self.use_dc:
        #     dc_out = F.relu(self.dc_projection(self.dc_in.permute(1, 0)))
        #     dec_out = dec_out + dc_out 
            
                                           

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))                  

        return dec_out


    def forward(self, x_enc, x_refer, mask=None):
        dec_out = self.forecast(x_enc, x_refer)
        return dec_out  # [B, L, D]
    
    