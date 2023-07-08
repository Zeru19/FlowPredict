import os
import math

import numpy as np
from tqdm import tqdm
import torch
from torch import optim
from torch import nn

from utils import regression_metrics, next_batch


def train(model, train_loader, val_loader, batch_size=64, shuffle=False,
          epoch_num=100, device=torch.device('cpu'), max_output_len=None,
          patience=0, init_lr=0.001, model_output='./model_params.pth'):

    optimizer = optim.Adam(model.parameters(), init_lr)
    criterion = nn.L1Loss()
    def _train_epoch(input_loader, validate=False):
        score_log = []
        if validate:
            model.eval()
        else:
            model.train()
        pbar = tqdm(enumerate(next_batch(input_loader, batch_size=batch_size, shuffle=shuffle)),
                    total=math.ceil(len(input_loader[0]) / batch_size))
        for i, (batch_x, batch_y, mask) in pbar:
            batch_x = batch_x.to(device)  # (bs, num_node, seq_len, 1)
            batch_y = batch_y.to(device)
            if mask is not None:
                mask = mask.to(device)
            # batch_x = batch_x.permute(0, 3, 2, 1)
            # pred, rep = model(batch_x, max_output_len)  #
            pred = model(batch_x, None, mask)  #
            # print("pred: ", pred.shape)
            # print("batch_y: ", batch_y.shape)
            loss = criterion(pred, batch_y)
            score_log.append(loss.item())
            pbar.set_description(
                "Batch %03d | loss: %.6f " % (i, loss.item())
            )
            
            if not validate:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return float(np.mean(score_log))

    min_loss, worse_round = 1e8, 0
    model_cache = './current_best_model.pth'
    for epoch in range(epoch_num):
        train_loss = _train_epoch(train_loader)
        val_loss = _train_epoch(val_loader, validate=True)
        print('Epoch %03d, train loss: %.6f, val loss: %.6f' % (epoch, train_loss, val_loss), flush=True)

        if patience > 0 and val_loss < min_loss:
            min_loss = val_loss
            worse_round = 0
            torch.save(model.state_dict(), model_cache)
        else:
            worse_round += 1

        if 0 < patience <= worse_round:
            print("Early stop @ epoch %d" % (epoch - worse_round), flush=True)
            break
        # flow.save(model.state_dict(), model_output)
        # print("Epoch %03d, model saved" % epoch)

    if patience > 0:
        model.load_state_dict(torch.load(model_cache))
        os.remove(model_cache)
    torch.save(model.cpu().state_dict(), model_output)


def test(model, test_loader, model_params, batch_size=64, device='cpu', max_seq_length=None):
    model.load_state_dict(torch.load(model_params))
    model.eval()
    y_pred = []
    y_real = []
    for batch_x, batch_y, mask in next_batch(test_loader, batch_size=batch_size):
        batch_x = batch_x.to(device)  # (bs, num_node, seq_len, 1)
        batch_y = batch_y.to(device)
        if mask is not None:
            mask = mask.to(device)
        # batch_x = batch_x.permute(0, 3, 2, 1)
        # pred, rep = model(batch_x, max_output_len)  #
        pred = model(batch_x, None, mask)
        y_pred.append(pred)
        y_real.append(batch_y)
        # print("pred: ", pred.shape)
    y_pred = torch.cat(y_pred, dim=0)
    y_real = torch.cat(y_real, dim=0)
    for t in range(y_real.shape[1]):
        mae, rmse, mape = regression_metrics(y_pred[:, t, ...], y_real[:, t, ...], 0.)
        print("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
            t + 1, mae, rmse, mape * 100))
    mae, rmse, mape = regression_metrics(y_pred, y_real, 0.)
    print("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
        mae, rmse, mape * 100))
