import torch
from torch import nn

from utils import weight_init
from models import BaseModel


class RnnPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super().__init__()

        self.encoder = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.decoder = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.out_linear = nn.Linear(hidden_size, output_size)

        self.apply(weight_init)

    def forward(self, input_seq, max_output_len):
        # input seq: (batch_size, max_len, input_size)
        encoder_out, encoder_hidden = self.encoder(input_seq)

        decoder_outs = []
        decoder_hidden = encoder_hidden
        decoder_x = encoder_out[:, -1:, :]  # (batch_size, 1, input_size)
        for step in range(max_output_len):
            decoder_out, decoder_hidden = self.decoder(decoder_x, decoder_hidden)  # (batch_size, 1, hidden_size)
            decoder_x = decoder_out.clone()
            decoder_out = self.out_linear(decoder_out)  # (batch_size, 1, output_size)

            decoder_outs.append(decoder_out)
        decoder_outs = torch.cat(decoder_outs, 1)  # (batch_size, max_output_len, output_size)
        return decoder_outs
    

class RNNModel(BaseModel):
    def __init__(self, lag, horizon, d_model, input_size, output_size, normalizer, denormalizer, 
                 num_layers=2, hidden_size=128, dropout=0.1):
        super(RNNModel, self).__init__(normalizer, denormalizer,
                                       'RNNModel-' + 
                                       f'lag{lag}-hor{horizon}-' +
                                       f'd{d_model}-h{hidden_size}-l{num_layers}')
        
        self.horizon = horizon
        self.d_model = d_model
        self.d_model = self.input_size = input_size
        self.output_size = output_size

        self.encoder = RnnPredictor(self.d_model, hidden_size, self.d_model,
                                    num_layers=num_layers, dropout=dropout)
        
    def forward(self, x, mark, mask):
        x = self.normalizer(x)
        x = self.encoder(x, self.horizon)
        x = self.denormalizer(x)

        return x
    