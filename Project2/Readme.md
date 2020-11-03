# Mini deep-learning framework built by pytorch
| Student's name | SCIPER |
| -------------- | ------ |
| Wei Jiang | 313794  |
| Minghui Shi | 308209 |
| Xiaoqi Ma |  308932 |

## Project Description
The objective of this project is to design a mini “deep learning framework” using only PyTorch’s
tensor operations and the standard math library, hence in particular without using autograd or the
neural-network(nn) modules.


### Folders and Files
- `model`:
  - `layers.py`: contains Linear for fully connected layer, Relu, Tanh, Leaky Relu, Exponential Relu and Sigmoid activation function layers.
  - `loss_func.py`: contains MSE loss function
  - `optimizers.py`: contains SGD, momentum SGD, Adagrad, Adam optmizers
  - `Module.py`: contains model structure for layers
  - `Sequential.py`: contains Sequential function to connect different layers
- `helpers.py`: contains helper functions for cross validation, training and testing
- `test.py`: run to train and test model with three hidden layers of 25 units using SGD as optimizer, MES loss as loss function and Relu, Tanh as activation function.  


## Getting Started
- Run `test.py` to train and test the model with 50 epochs. Cross validation is automatically run to find the best learning rate corresponding to the maximal validation accuracy. The MSE loss, training and test errors are logged for each epoch. Final training and test errors are printed at the end.
