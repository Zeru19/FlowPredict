import torch

from dataset import Dataset
from models import RNNModel, TransformerModel
from trainer import train, test


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 16

data = Dataset('pems08')

mask_pre = False
train_iter = data.get_samples('train', mask_pre=mask_pre)
val_iter = data.get_samples('val', mask_pre=mask_pre)
test_iter = data.get_samples('test', mask_pre=mask_pre)
normalizer, denormalizer = data.get_normalizer()

model = RNNModel(lag=data.lag,
                 horizon=data.horizon,
                 d_model=data.num_nodes,
                 input_size=data.num_nodes,
                 output_size=data.num_nodes,
                 normalizer=normalizer,
                 denormalizer=denormalizer,
                 hidden_size=256).to(device)

# model = TransformerModel(lag=data.lag,
#                         horizon=data.horizon,
#                         d_model=128,
#                         nhead=8,
#                         input_size=data.num_nodes,
#                         output_size=data.num_nodes,
#                         normalizer=normalizer,
#                         denormalizer=denormalizer,
#                         hidden_size=256).to(device)

train(model, train_iter, val_iter,
      batch_size=64, shuffle=True, epoch_num=100,
      patience=10, init_lr=0.001, device=device,
      model_output='params/{}.pth'.format(model.name))

test(model, test_iter,
     batch_size=64,
     model_params='params/{}.pth'.format(model.name))
