import torch
from matplotlib import pyplot as plt
import math
from torch import empty

# set seed for data generation
from tqdm import tqdm


def generate_disc_set(nb):
    """
    generate data and label
    :param nb: number of data points to generate
    :return pts: generated data points
    :return label: generated labels
    """
    pts = empty(nb, 2).uniform_(0, 1)
    label = (-((pts - 0.5).pow(2).sum(1) - (1 / (
                2 * math.pi))).sign() + 1) / 2  # take reversed sign of subtraction with 2/pi then add 1 and divide by 2 then reverse
    return pts, label


def one_hot_encoding(label):
    """
    convert data label to one-hot encoding format
    :param label: input label to be converted
    :return: generated one-hot encoding target
    """
    target = torch.zeros(len(label), 2)
    for i in range(len(label)):
        if label[i] == 0:
            target[i][0] = 1
        else:
            target[i][1] = 1
    return target


def normalize(data):
    """
    normalize data to get mean =0 and std =1
    :param data: data to be normalized
    :return: normalized data
    """
    mean_x = data.mean()
    std_x = data.std()
    data = (data - mean_x) / std_x
    return data




def calc_accuracy(model, input, target):
    """
    Evaluate given network model with given input set and target
    :param model: network model to be evaluated
    :param input: data to be predicted
    :param target: the ground truth
    :return: the accuracy of the prediction with respect to ground truth
    """
    target = torch.argmax(target, 1)
    total = len(target)
    output = model.forward(input)
    _, pred = torch.max(output, 1)
    correct = (pred == target).sum().item()
    return correct / total


def train(model, Loss, optimizer, input_tr, target_tr, input_te, target_te, nb_epochs, batch_size):
    """
    Train network model with given parameters
    :param model: network model to be trained
    :param loss: loss for training
    :param optimizer: optimizer for training
    :param input_tr: the input training data
    :param target_tr: the ground truth of training data
    :param input_te: the input testing data
    :param target_te: the ground truth of testing data
    :param nb_epochs: number of training epochs
    :return acc_tr: final training accuracy
    :return acc_te: final testing accuracy
    """
    for e in range(nb_epochs):
        # shuffle the training data in each epoch
        # indices = torch.randperm(input_tr.shape[0])
        # input_tr = input_tr[indices]
        # target_tr = target_tr[indices]
        with tqdm(total = input_tr.shape[0], desc=f'Epoch {e + 1}/{nb_epochs}', unit='pts') as pbar:
            for b in range(0, input_tr.shape[0], batch_size):
                output = model.forward(input_tr.narrow(0, b, batch_size))
                model.zero_grad()
                tmp = Loss.backward(output, target_tr.narrow(0, b, batch_size))
                model.backward(tmp)
                optimizer.update()
                pbar.update(batch_size)
            output_tr = model.forward(input_tr)
            loss_e_tr = Loss.forward(output_tr, target_tr).item()
            output_te = model.forward(input_te)
            loss_e_te = Loss.forward(output_te, target_te).item()
            acc_tr = calc_accuracy(model, input_tr, target_tr)
            acc_te = calc_accuracy(model, input_te, target_te)
            pbar.set_postfix(**{'train loss': loss_e_tr, 'test loss': loss_e_te, "train error": 1 - acc_tr,
                                "test error": 1- acc_te})
    return  acc_tr, acc_te


def cross_validation(model, optimizer_name, nb_epochs, batch_size, loss, k_fold, lr_set, input, target):
    """
    Perform K-fold cross validation to optimize hyperprameters lr, reg, and/or gamma on the given model with given parameters
    :param model: network model
    :param optimizer_name: name of optimizer to be used
    :param nb_epochs: number of epochs
    :param batch_size: data batch size
    :param loss: loss to be used for training
    :param k_fold: number of cross validation folds
    :param lr_set: set of lr
    :param input: input data for cross validation
    :param target: label of input data
    :return best_lr.item(): best learning rate for training
    :return loss_tr_set: training loss
    :return loss_te_set: testing loss
    :return acc_tr_set: accuracy of training set
    :return acc_te_set: accuracy of testing set
    """
    interval = int(input.shape[0] / k_fold)
    indices = torch.randperm(input.shape[0])
    loss_tr_set = []
    loss_te_set = []
    acc_tr_set = []
    acc_te_set = []
    max_acc_te = 0.
    best_lr = 0
    for i, lr in enumerate(lr_set):
        loss_tr = 0
        loss_te = 0
        acc_tr = 0
        acc_te = 0
        print("Running cross validation. Progress:  ", i / len(lr_set) * 100, '%')

        for k in range(k_fold):
            model.reset()
            optimizer = optimizer_name(model=model, lr=lr.item())
            train_indices = indices[k * interval:(k + 1) * interval]
            input_te = input[train_indices]
            target_te = target[train_indices]
            residual = torch.cat((indices[0:k * interval], indices[(k + 1) * interval:]), 0)
            input_tr = input[residual]
            target_tr = target[residual]
            loss_tr_temp, loss_te_temp, acc_tr_temp, acc_te_temp = train_cv(model, loss, optimizer, input_tr, target_tr,
                                                                         input_te, target_te, nb_epochs=nb_epochs,
                                                                         batch_size=batch_size)
            loss_tr += loss_tr_temp[-1]
            loss_te += loss_te_temp[-1]
            acc_tr += acc_tr_temp[-1]
            acc_te += acc_te_temp[-1]

        loss_tr_set.append(loss_tr / k_fold)
        loss_te_set.append(loss_te / k_fold)
        acc_tr_set.append(acc_tr / k_fold)
        acc_te_set.append(acc_te / k_fold)

        if acc_te_set[-1] > max_acc_te:
            max_acc_te = acc_te_set[-1]
            best_lr = lr

    return best_lr.item()

def train_cv(model, Loss, optimizer, input_tr, target_tr, input_te, target_te, nb_epochs, batch_size):
    """
    Train network model with given parameters
    :param model: network model to be trained
    :param loss: loss for training
    :param optimizer: optimizer for training
    :param input_tr: the input training data
    :param target_tr: the ground truth of training data
    :param input_te: the input testing data
    :param target_te: the ground truth of testing data
    :param nb_epochs: number of training epochs
    :return loss_history_tr: training loss list for each epoch
    :return loss_history_te: testing loss list for each epoch
    :return acc_history_tr: training accuracy list for each epoch
    :return acc_history_te: testing accuracy list for each epoch
    """
    loss_history_tr = []
    loss_history_te = []
    acc_history_tr = []
    acc_history_te = []
    for e in range(nb_epochs):
            for b in range(0, input_tr.shape[0], batch_size):
                output = model.forward(input_tr.narrow(0, b, batch_size))
                model.zero_grad()
                tmp = Loss.backward(output, target_tr.narrow(0, b, batch_size))
                model.backward(tmp)
                optimizer.update()
            output_tr = model.forward(input_tr)
            loss_e_tr = Loss.forward(output_tr, target_tr).item()
            output_te = model.forward(input_te)
            loss_e_te = Loss.forward(output_te, target_te).item()
            loss_history_tr.append(loss_e_tr)
            loss_history_te.append(loss_e_te)
            acc_history_tr.append(calc_accuracy(model, input_tr, target_tr))
            acc_history_te.append(calc_accuracy(model, input_te, target_te))

    return loss_history_tr, loss_history_te, acc_history_tr, acc_history_te


