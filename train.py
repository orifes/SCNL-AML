from torch import nn, optim
from models.models import RTModel, GRUNet
from pair_word_ewe import pair_word_to_ewe
import torch
MAX_RATE_VALUE = 100
HIDDEN_DIM = 100

OUTPUT_SIZE = 1

AB_INPUT_SIZE = 768
FT_INPUT_SIZE = 300


def get_tensors_for_video(rating_file_path, input_file_path):
    res_df = pair_word_to_ewe(rating_file_path, input_file_path)
    x_ab = torch.stack([vec.cuda().reshape((vec.shape[0], vec.shape[-1])) for vec in res_df.ab_embedding.values])
    x_ft = torch.stack([torch.tensor(vec) for vec in res_df.fasttext_vector.values]).cuda()
    y = torch.tensor([rat / MAX_RATE_VALUE for rat in res_df.rating.values]).cuda()
    return x_ab, x_ft, y


def train_rt(model, X, Y, epochs, optimizer, criterion):
    for i in range(1, epochs + 1):
        data_num = 0
        loss_sum = 0.0
        for j, x, y_true in enumerate(zip(X, Y)):
            optimizer.zero_grad()
            y_pred = model(x).squeeze()
            loss = criterion(y_pred, y_true)
            loss_sum += loss
            data_num += x.shape[1]
            loss.backward()
            optimizer.step()
        print("Epoch {}: average loss {}".format(i, loss_sum/data_num))


def test(model, criterion, X, Y):
    predictions = []
    data_num = 0
    loss_sum, corr, ccc = 0.0, [], []
    for j, x, y_true in enumerate(zip(X, Y)):
        y_pred = model(x).squeeze()
        loss = criterion(y_pred, y_true)
        predictions.append(y_pred)
        loss_sum += loss
        data_num += x.shape[1]


def train_gru(model, X, Y, epochs, optimizer, criterion):
    for i in range(1, epochs + 1):
        data_num = 0
        loss_sum = 0.0
        for j, x, y_true in enumerate(zip(X, Y)):
            h = model.init_h(1)
            for k, word_vec, word_y in enumerate(zip(x,y_true)):
                optimizer.zero_grad()
                y_pred, h = model(word_vec[None, :, :], h).squeeze()
                loss = criterion(y_pred, word_y)
                loss_sum += loss
                data_num += 1
                loss.backward()
                optimizer.step()
        print("Epoch {}: average loss {}".format(i, loss_sum/data_num))