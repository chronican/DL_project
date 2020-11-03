import logging
import torch
from Project2.model.layers import Linear, Relu, Tanh
from Project2.model.loss_func import MSELoss
from Project2.model.optimizers import SGD
from Project2.model.Sequential import Sequential
from Project2.helpers import normalize, generate_disc_set, train, cross_validation, one_hot_encoding

torch.manual_seed(0)
# Generate training and test data sets and normalize
train_input, train_label = generate_disc_set(1000)
test_input, test_label = generate_disc_set(1000)
train_target = one_hot_encoding(train_label)
test_target = one_hot_encoding(test_label)
train_input = normalize(train_input)
test_input = normalize(test_input)

# K-fold cross validation to optimize learning rate over range lr_set
lr_set = torch.logspace(-3, -0.1, 5)
k_fold = 5

# set loss and optimizer
nb_epochs = 50
batch_size = 50
loss = MSELoss()
optimizer_name = SGD

# create models
model = Sequential(Linear(2, 25), Relu(), Linear(25, 25), Relu(), Linear(25, 25), Relu(), Linear(25, 2), Tanh())

# cross validation to find best learning rate
print("cross validation to get best leaning rate for model")
best_lr = cross_validation(model, optimizer_name, nb_epochs, batch_size,
                           loss, k_fold, lr_set, train_input, train_target)

# initialize models
optimizer = optimizer_name(model=model, lr=best_lr)
# logging info
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logging.info(f'''Starting training:
    Epochs:          {nb_epochs}
    Learning rate:   {best_lr}
    Batch size:      {batch_size}
    Optimizer:       {"SGD"}
    Loss:            {"MSE loss"}
    Training size:   {1000}
    Test size:       {1000}
''')

# model training
acc_train, acc_test = train(model, loss, optimizer, train_input, train_target, test_input, test_target,
                            nb_epochs=nb_epochs, batch_size=batch_size)

# print error rate
print("Final training error:", 1 - acc_train)
print("Final testing error:", 1 - acc_test)
