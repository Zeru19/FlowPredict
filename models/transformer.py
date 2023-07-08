import torch
from torch import nn
from einops import repeat

from utils import weight_init
from models import BaseModel
from models.s4d import TimeEncoder


class TransformerModel(BaseModel):
    def __init__(self, lag, horizon, d_model, input_size, output_size, normalizer, denormalizer, 
                 num_layers=2, nhead=8, hidden_size=128, dropout=0.1):
        
        super(TransformerModel, self).__init__(normalizer, denormalizer,
                                               'Transformer-' +
                                               f'lag{lag}-hor{horizon}-' +
                                               f'd{d_model}-h{hidden_size}-l{num_layers}-h{nhead}')
        self.horizon = horizon
        self.d_model = d_model
        self.input_size = input_size
        self.output_size = output_size

        self.time_embed = TimeEncoder(d_model)
        self.pre_linear = nn.Linear(input_size, self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, hidden_size, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.out_linear = nn.Linear(self.d_model, output_size)

    def forward(self, x, mark, mask):
        x = self.normalizer(x)
        x = self.pre_linear(x)

        x = x * repeat(1 - mask, 'b l -> b l h', h=self.d_model)
        x = self.time_embed(x, mark, mask)
        x = self.encoder(x)
        x = self.out_linear(x)
        x = self.denormalizer(x)
        
        return x[:, -self.horizon:, :]