import torch

from dataset import Dataset
from models import S4DModel
from trainer import train, test


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64

data = Dataset('pems08')

train_iter = data.get_samples('train', mask_pre=True)
val_iter = data.get_samples('val', mask_pre=True)
test_iter = data.get_samples('test', mask_pre=True)
normalizer, denormalizer = data.get_normalizer()

model = S4DModel(lag=data.lag,
                 horizon=data.horizon,
                 d_model=data.num_nodes,
                 input_size=data.num_nodes,
                 output_size=data.num_nodes,
                 normalizer=normalizer,
                 denormalizer=denormalizer,
                 d_state=256,
                 hidden_size=256).to(device)

# train(model, train_iter, val_iter,
#       batch_size=64, shuffle=True, epoch_num=100,
#       patience=10, init_lr=0.001, device=device,
#       model_output='params/{}.pth'.format(model.name))

test(model, test_iter,
     batch_size=64,
     model_params='params/{}.pth'.format(model.name), 
     device=device)
