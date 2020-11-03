import torch
from torch import nn
from models.Siamese import Siamese
from models.CNN import CNN
from models.FNN import FNN
from models.Resnet import ResNet
from helpers import get_train_stats, cross_validation
from dataset import generate_pair_sets
from torch.utils.data import DataLoader

# Initial setups
# set seed
torch.manual_seed(0)
# boolean flag: True to run cross validation; False to run training and testing
run_cross_validation = False

# generate train and test sets
N = 1000
train_input, train_target, train_class, test_input, test_target, test_class = generate_pair_sets(N)

# Data loaders
batch_size = 100
train_loader = DataLoader(list(zip(train_input, train_target, train_class)), batch_size)
test_loader = DataLoader(list(zip(test_input, test_target, test_class)), batch_size)
nb_channels = 2  # input channel
nb_digits = 10  # number of digit classes
nb_class = 2  # number of output classes

# loss function
cross_entropy = nn.CrossEntropyLoss()

epochs = 25  # number of epochs
trial = 11
# cases with/without weight sharing and auxiliary loss
weight_sharing = [False, True, False]
auxiliary_loss = [False, False, True]
# auxiliary loss weighting

mean_tr = []
mean_te = []
std_tr = []
std_te = []

#FNN: Run cross validation or training and testing
print("-- Training for FNN Model--\n")
AL_weight = 0.4
model = FNN
if run_cross_validation:  # Run cross validation to help select optimal hyperparameter
    k_fold = 5
    lr_set = [0.0001, 0.001, 0.01, 0.1]  # learning rate range for cross validation
    reg_set = [0, 0.1, 0.2, 0.3]  # weight decay factor range
    gamma_set = [0, 0.1]  # learning rate scheduler multiplicative factor range
    for i in range(len(auxiliary_loss)):
        cross_validation(k_fold, lr_set, reg_set, gamma_set, model, cross_entropy, AL_weight, epochs,
                         batch_size=batch_size, weight_sharing=weight_sharing[i], auxiliary_loss=auxiliary_loss[i])

#train and test the model
#hyperparameters for training and testing
reg = [0.05, 0.03, 0.06]  # weight decay factor
lr = [0.003, 0.005, 0.003]  # learning rate
gamma = [0, 0, 0]  # learing rate scheduler's multiplicative factor

for i in range(len(auxiliary_loss)):
    mean_acc_tr, std_acc_tr, mean_acc_te, std_acc_te = get_train_stats(model, lr[i], reg[i], cross_entropy,
                                                                       AL_weight=AL_weight, trial=trial, epochs=epochs,
                                                                       gamma=gamma[i], weight_sharing=weight_sharing[i],
                                                                       auxiliary_loss=auxiliary_loss[i])
    mean_tr.append(mean_acc_tr)
    mean_te.append(mean_acc_te)
    std_tr.append(std_acc_tr)
    std_te.append(std_acc_te)

#print the test results
print("-- Result for FNN Model--\n")
for j in range(len(auxiliary_loss)):
    print("\n Auxiliary loss: ", auxiliary_loss[j], ", weight sharing", weight_sharing[j],
          ", Train Accuracy: Mean = %.2f" % mean_tr[j], ", STD = %.2f" % std_tr[j],
          ", Test Accuracy: Mean = %.2f" % mean_te[j], "STD = %.2f" % std_te[j])

######################################################################################################################
#Siamese: Run cross validation or training and testing

print("-- Training for SiameseNet Model--\n")
print()
AL_weight = 1
model = Siamese
mean_tr = []
mean_te = []
std_tr = []
std_te = []
if run_cross_validation:  # Run cross validation to help select optimal hyperparameter
    k_fold = 5
    lr_set = [0.0001, 0.001, 0.01, 0.1]  # learning rate range for cross validation
    reg_set = [0, 0.1, 0.2, 0.3]  # weight decay factor range
    gamma_set = [0, 0.1]  # learning rate scheduler multiplicative factor range
    for i in range(len(auxiliary_loss)):
        cross_validation(k_fold, lr_set, reg_set, gamma_set, model, cross_entropy, AL_weight, epochs,
                         batch_size=batch_size, weight_sharing=weight_sharing[i], auxiliary_loss=auxiliary_loss[i])

# train and test the model
# hyperparameters for training and testing
reg = [0.001, 0.001, 0.001]  # weight decay factor
lr = [0.01, 0.01, 0.01]  # learning rate
gamma = [0, 0, 0]  # learing rate scheduler's multiplicative factor

for i in range(len(auxiliary_loss)):
    mean_acc_tr, std_acc_tr, mean_acc_te, std_acc_te = get_train_stats(model, lr[i], reg[i], cross_entropy,
                                                                       AL_weight=AL_weight, trial=trial, epochs=epochs,
                                                                       gamma=gamma[i], weight_sharing=weight_sharing[i],
                                                                       auxiliary_loss=auxiliary_loss[i])
    mean_tr.append(mean_acc_tr)
    mean_te.append(mean_acc_te)
    std_tr.append(std_acc_tr)
    std_te.append(std_acc_te)

# print the test results
for j in range(len(auxiliary_loss)):
    print("-- Result for SiameseNet Model--\n")
    print("Auxiliary loss: ", auxiliary_loss[j], ", weight sharing", weight_sharing[j],
          ", Train Accuracy: Mean = %.2f" % mean_tr[j], ", STD = %.2f" % std_tr[j],
          ", Test Accuracy: Mean = %.2f" % mean_te[j], "STD = %.2f" % std_te[j])

########################################################################################################################
# CNN: Run cross validation or training and testing
# print("CNN Model")
# model = CNN
#

print("-- Training for CNN Model--\n")
AL_weight = 1
model = CNN
mean_tr = []
mean_te = []
std_tr = []
std_te = []
if run_cross_validation:  # Run cross validation to help select optimal hyperparameter
    k_fold = 5
    lr_set = [0.0001, 0.001, 0.01, 0.1]  # learning rate range for cross validation
    reg_set = [0, 0.1, 0.2, 0.3]  # weight decay factor range
    gamma_set = [0, 0.1]  # learning rate scheduler multiplicative factor range
    for i in range(len(auxiliary_loss)):
        cross_validation(k_fold, lr_set, reg_set, gamma_set, model, cross_entropy, AL_weight, epochs,
                         batch_size=batch_size, weight_sharing=weight_sharing[i], auxiliary_loss=auxiliary_loss[i])

# train and test the model
# hyperparameters for training and testing
reg = [0.15, 0.1, 0.3]  # weight decay factor
lr = [0.0015, 0.0015, 0.0025]  # learning rate
gamma = [0.2, 0.1, 0.1]  # learing rate scheduler's multiplicative factor

for i in range(len(auxiliary_loss)):
    mean_acc_tr, std_acc_tr, mean_acc_te, std_acc_te = get_train_stats(model, lr[i], reg[i], cross_entropy,
                                                                       AL_weight=AL_weight, trial=trial, epochs=epochs,
                                                                       gamma=gamma[i], weight_sharing=weight_sharing[i],
                                                                       auxiliary_loss=auxiliary_loss[i])
    mean_tr.append(mean_acc_tr)
    mean_te.append(mean_acc_te)
    std_tr.append(std_acc_tr)
    std_te.append(std_acc_te)

# print the test results
print("-- Result for CNN Model--\n")

for j in range(len(auxiliary_loss)):
    print("Auxiliary loss: ", auxiliary_loss[j], ", weight sharing", weight_sharing[j],
          ", Train Accuracy: Mean = %.2f" % mean_tr[j], ", STD = %.2f" % std_tr[j],
          ", Test Accuracy: Mean = %.2f" % mean_te[j], "STD = %.2f" % std_te[j])

########################################################################################################################
# ResNet: Run cross validation or training and testing

print("-- Training for Resnet Model--\n")
AL_weight = 0.5
model = ResNet
mean_tr = []
mean_te = []
std_tr = []
std_te = []
if run_cross_validation:  # Run cross validation to help select optimal hyperparameter
    k_fold = 5
    lr_set = [0.0001, 0.001, 0.01, 0.1]  # learning rate range for cross validation
    reg_set = [0, 0.1, 0.2, 0.3]  # weight decay factor range
    gamma_set = [0, 0.1]  # learning rate scheduler multiplicative factor range

    for i in range(len(auxiliary_loss)):
        cross_validation(k_fold, lr_set, reg_set, gamma_set, model, cross_entropy, AL_weight, epochs,
                         batch_size=batch_size, weight_sharing=weight_sharing[i], auxiliary_loss=auxiliary_loss[i])

# train and test the model
# hyperparameters for training and testing
reg = [0.001, 0.002, 0.001]  # weight decay factor
lr = [0.0035, 0.0035, 0.0025]  # learning rate
gamma = [0.1, 0.2, 0.1]  # learing rate scheduler's multiplicative factor

for i in range(len(auxiliary_loss)):
    mean_acc_tr, std_acc_tr, mean_acc_te, std_acc_te = get_train_stats(model, lr[i], reg[i], cross_entropy,
                                                                       AL_weight=AL_weight, trial=trial, epochs=epochs,
                                                                       gamma=gamma[i], weight_sharing=weight_sharing[i],
                                                                       auxiliary_loss=auxiliary_loss[i])
    mean_tr.append(mean_acc_tr)
    mean_te.append(mean_acc_te)
    std_tr.append(std_acc_tr)
    std_te.append(std_acc_te)

# print result
print("-- Result for ResNet Model--\n")
for j in range(len(auxiliary_loss)):
    print("Auxiliary loss: ", auxiliary_loss[j], ", weight sharing", weight_sharing[j],
          ", Train Accuracy: Mean = %.2f" % mean_tr[j], ", STD = %.2f" % std_tr[j],
          ", Test Accuracy: Mean = %.2f" % mean_te[j], "STD = %.2f" % std_te[j])
