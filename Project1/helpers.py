import logging
import torch
from torch import optim
from tqdm import tqdm
from dataset import generate_pair_sets
from torch.utils.data import DataLoader
import numpy as np




def evaluate(model, data_loader, auxiliary_loss, criterion):
    """
    Evaluate given network model with given data set and parameters
    :param model: network model to be evaluated
    :param data_loader: data loader that contains image, target, and digit_target
    :param auxiliary_loss: boolean flag for applying auxiliary loss
    :param criterion: loss function
    :return correct/total: primary task accuracy
    :return correct_digit/2/total: average digit recognition accuracy
    :return loss: testing loss
    """
    correct = 0
    correct_digit = 0
    total = 0
    loss = 0
    for (image, target, digit_target) in data_loader:
        total += len(target)
        if not auxiliary_loss: # case without auxiliary loss
            output = model(image)
            loss += criterion(output, target)
            _, pred = torch.max(output, 1)
            correct += (pred == target).sum().item()
        else: # case with auxiliary loss
            digit1, digit2, output = model(image)
            loss += criterion(output, target)
            _, pred = torch.max(output, 1)
            correct += (pred == target).sum().item()
            _, pred1 = torch.max(digit1, 1)
            correct_digit += (pred1 == digit_target[:, 0]).sum().item()
            _, pred2 = torch.max(digit2, 1)
            correct_digit += (pred2 == digit_target[:, 1]).sum().item()
    if not auxiliary_loss:
        return correct / total, loss
    else:
        return correct / total, correct_digit / 2 / total, loss


def train(train_data_loader, test_data_loader,
          model, optimizer, criterion, pbar = None, AL_weight=0.5,
          epochs=25,  gamma = 0, weight_sharing=False, auxiliary_loss=False):
    """
    Train network model with given parameters
    :param train_data_loader: data loader for training set
    :param test_data_loader: data load for test(validation set)
    :param model: network model to be trained
    :param optimizer: optimizer for training
    :param criterion: loss function for optimizer
    :param pbar: Progress bar for logging
    :param AL_weight: Weight applied to auxiliary loss when combined with primary task loss
    :param epochs: number of training epochs
    :param gamma: learning rate scheduler's multiplicative factor
    :param weight_sharing:  boolean flag for applying weight sharing
    :param auxiliary_loss:  boolean flag for applying auxiliary loss
    :return acc_train: primary task accuracies of training set
    :return acc_test: primary task accuracies of testing set
    :return acc_train_digit: digit recognition accuracies of traing set
    :return acc_test_digit: digit recognition accuracies of test set
    :return loss_tr: training loss
    :return loss_te: test(validation) loss
    """
    if gamma != 0: # if gamma is not 0, set up learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=gamma)

    for epoch in range(epochs):
            for (image, target, digit_target) in train_data_loader:
                model.train()
                optimizer.zero_grad()
                if auxiliary_loss: # case with auxiliary loss
                    digit1, digit2, output = model(image)
                    loss = criterion(output, target) # primary task loss
                    loss += AL_weight * criterion(digit1, digit_target[:, 0]) # add weighted aux loss from 1st digit
                    loss += AL_weight * criterion(digit2, digit_target[:, 1]) # add weighted aux loss from 2nd digit
                else: # case without auxiliary loss
                    output = model(image)
                    loss = criterion(output, target) # primary task loss
                loss.backward()
                optimizer.step()
                if gamma != 0 and epoch > 5:
                    scheduler.step()
                if pbar:
                    pbar.set_postfix(**{'loss (batch)': loss.item()})
                    pbar.update(target.shape[0])
    # evaluate model at the end of last epoch
    model.eval()
    with torch.no_grad():
        if auxiliary_loss: # case with aux loss
            acc_train, acc_train_digit, loss_tr = evaluate(model, train_data_loader, auxiliary_loss, criterion) # evaluate model with training set
            acc_test, acc_test_digit, loss_te = evaluate(model, test_data_loader, auxiliary_loss, criterion) # evaluate model with test set
        else: # case without aux loss
            acc_train, loss_tr= evaluate(model, train_data_loader, auxiliary_loss, criterion)# evaluate model with training set
            acc_test, loss_te = evaluate(model, test_data_loader, auxiliary_loss, criterion) # evaluate model with test set
    if auxiliary_loss:
        return acc_train, acc_test, acc_train_digit, acc_test_digit, loss_tr, loss_te
    else:
        return acc_train, acc_test, loss_tr, loss_te

def get_train_stats(model, lr, reg, criterion, AL_weight, epochs, trial, batch_size = 100, gamma = 0, weight_sharing = False, auxiliary_loss = False):
    """
    Perform training and testing for a given number of trials with the given model and parameters
    :param model: network model
    :param lr: learning rate of the model
    :param reg: weight decay parameter of the model
    :param criterion: loss function
    :param AL_weight: Weight applied to auxiliary loss when combined with primary task loss
    :param epochs: number of training epochs
    :param trial: number of trials
    :param batch_size: data batch size
    :param test_every: number of steps between each model evaluation
    :param gamma: learning rate shceduler's multiplicative factor
    :param weight_sharing: boolean flag for applying weight sharing
    :param auxiliary_loss: boolean flag for applying auxiliary loss
    :return np.mean(accuracy_trial_tr): mean primary task training accuracy across trials
    :return np.std(accuracy_trial_tr): standard deviation of primary task training accuracy across trials
    :return np.mean(accuracy_trial_te): mean primary task test accuracy across trials
    :return np.std(accuracy_trial_te): standard deviation of primary task test accuracy across trials
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'''Starting training:
        Trials:          {trial}
        Epochs:          {epochs}
        Learning rate:   {lr}
        Batch size:      {100}
        Training size:   {1000}
        Test size:       {1000}
        Weight Sharing:  {weight_sharing}
        Auxialiary Loss:  {auxiliary_loss}
        Auxialiary loss weight: {AL_weight}
    ''')
    accuracy_trial_tr = []
    accuracy_trial_te = []
    mean_acc_tr = []
    mean_acc_te = []
    mean_acc_aux_tr = []
    mean_acc_aux_te = []
    mean_losses_tr = []
    mean_losses_te = []
    nb_channels = 2
    nb_class = 2
    for i in range(trial):
        with tqdm(total=25000, desc=f'Trial {i + 1}/{trial}', unit='img') as pbar:
            net = model(nb_channels, nb_class, weight_sharing, auxiliary_loss)
            train_input, train_target, train_class, test_input, test_target, test_class = generate_pair_sets(1000)
            # Data loaders
            train_loader = DataLoader(list(zip(train_input, train_target, train_class)), batch_size, shuffle=True)
            test_loader = DataLoader(list(zip(test_input, test_target, test_class)), batch_size, shuffle=True)
            train_info = train(train_loader, test_loader,
                               model=net,
                               optimizer=optim.Adam(net.parameters(), lr=lr, weight_decay=reg),
                               criterion=criterion, pbar= pbar, AL_weight=AL_weight,
                               epochs=epochs,  gamma = gamma,
                               weight_sharing=weight_sharing,
                               auxiliary_loss=auxiliary_loss)
            if auxiliary_loss:
                accuracy_train, accuracy_test, acc_train_digit, acc_test_digit, losses_tr, losses_te = train_info
                mean_acc_aux_tr.append(acc_train_digit)
                mean_acc_aux_te.append(acc_test_digit)
            else:
                accuracy_train, accuracy_test, losses_tr, losses_te = train_info
            if auxiliary_loss: # log results
                pbar.set_postfix(**{"train loss ": losses_tr.item(), "Test loss": losses_te.item(),"train acccuracy": accuracy_train,
                            "test accuracy": accuracy_test,
                            "train digit accuracy ": acc_train_digit,
                            "test digit accuracy ": acc_test_digit})
            else:
                pbar.set_postfix(**{"train loss": losses_tr.item(), "test loss": losses_te.item(), "train acccuracy": accuracy_train,
                            "test accuracy": accuracy_test})
        mean_acc_tr.append(accuracy_train)
        mean_acc_te.append(accuracy_test)
        mean_losses_tr.append(losses_tr)
        mean_losses_te.append(losses_te)
        accuracy_trial_tr.append(accuracy_train)
        accuracy_trial_te.append(accuracy_test)

    return np.mean(accuracy_trial_tr), np.std(accuracy_trial_tr), np.mean(accuracy_trial_te), np.std(accuracy_trial_te)





def cross_validation(k_fold, lr_set, reg_set, gamma_set, model, criterion, AL_weight, epochs,
                     batch_size = 100,  weight_sharing = False, auxiliary_loss = False):
    """
    Perform K-fold cross validation to optimize hyperprameters lr, reg, and/or gamma on the given model with given parameters
    :param k_fold: number of cross validation folds
    :param lr_set: set of lr
    :param reg_set: set of reg
    :param gamma_set: set of gamma
    :param model: network model
    :param criterion: loss function
    :param AL_weight: Weight applied to auxiliary loss when combined with primary task loss
    :param epochs: number of epochs
    :param batch_size: data batch size
    :param weight_sharing: boolean flag for applying weight sharing
    :param auxiliary_loss: boolean flag for applying auxiliary loss
    :print best lr, best reg, best gamma (all based on maximum validation accuracy),
    set of training loss, set of validation loss, set of primary task training accuracy, set of primary task validation accuracy
    (all across hyperparameters tested)
    """
    nb_channels = 2
    nb_class = 2

    train_input, train_target, train_class, _, _, _ = generate_pair_sets(1000) # generate data set

    interval = int(train_input.shape[0]/ k_fold) # calculate data set size for each fold
    indices = torch.randperm(train_input.shape[0]) # shuffle data indicies

    accuracy_tr_set = []
    accuracy_te_set = []

    max_acc_te = 0
    counter = 0
    for lr in lr_set:
        for reg in reg_set:
            for gamma in gamma_set:
                accuracy_tr_k = 0
                accuracy_te_k = 0
                counter += 1
                print("Running cross validation. Progress:  ", counter/(len(reg_set)*len(lr_set)*len(gamma_set))*100, '%')

                print('lr = ', lr, 'reg = ', reg, 'gamma = ', gamma)
                for k in range(k_fold):
                    # initialize model
                    net = model(nb_channels, nb_class, weight_sharing, auxiliary_loss)
                    # divide data into k-fold and prepare train and validation sets
                    train_indices = indices[k*interval:(k+1)*interval]
                    input_te = train_input[train_indices]
                    target_te = train_target[train_indices]
                    digit_target_te = train_class[train_indices]
                    residual = torch.cat((indices[0:k*interval],indices[(k+1)*interval:]),0)
                    input_tr = train_input[residual]
                    target_tr = train_target[residual]
                    digit_target_tr = train_class[residual]

                    # Data loaders
                    train_loader = DataLoader(list(zip(input_tr, target_tr, digit_target_tr)), batch_size, shuffle=True)
                    test_loader = DataLoader(list(zip(input_te, target_te, digit_target_te)), batch_size, shuffle=True)
                    train_info = train(train_loader, test_loader,
                                       model=net,
                                       optimizer=optim.Adam(net.parameters(), lr=lr, weight_decay=reg),
                                       criterion=criterion, AL_weight=AL_weight,
                                       epochs=epochs, gamma = gamma,
                                       weight_sharing=weight_sharing,
                                       auxiliary_loss=auxiliary_loss)
                    if auxiliary_loss: # with auxiliary loss
                        accuracy_train, accuracy_test, acc_train_digit, acc_test_digit, losses_tr, losses_te = train_info
                    else: # no auxiliary loss
                        accuracy_train, accuracy_test, losses_tr, losses_te = train_info

                    accuracy_tr_k += accuracy_train
                    accuracy_te_k += accuracy_test

                accuracy_tr_set.append(accuracy_tr_k/k_fold)
                accuracy_te_set.append(accuracy_te_k/k_fold)

                if accuracy_te_set[-1] > max_acc_te:# compare current validation accuracy  with the current maximum
                    # update hyperparameters associated with max val accuracy and print the new max accuracy
                    max_acc_te = accuracy_te_set[-1]
                    best_lr = lr
                    best_reg = reg
                    best_gamma = gamma
                    print(f"Max val acc so far: {max_acc_te}")
    print(f"Best lr: {best_lr}, Best reg: {best_reg}, Best gamma: {best_gamma}, Max val acc: {max_acc_te}")
