# Weight sharing and auxiliary losses for classification
| Student's name | SCIPER |
| -------------- | ------ |
| Wei Jiang | 313794  |
| Minghui Shi | 308209 |
| Xiaoqi Ma | 308932  |

## Project Description
The objective of this project is to test different architectures to compare two digits visible in a two-channel image. It aims at showing in particular the impact of weight sharing, and of the use of an auxiliary loss to help the training of the main objective. We use four architectures - FNN, CNN, SiameseNet and ResNet to test their responses to weight sharing and auxiliary loss.

### Folders and Files
- `models`:
  - `FNN.py`: contains FNN model
  - `CNN.py`: contains CNN model
  - `ResNet.py`: contains ResNet model
  - `Siamese.py`: contains Siamese network model
- `dataset.py`: generate pair of images as our dataset for training and testing
- `helpers.py`: contains helper functions for training and testing
- `test.py`: run to train and test the four models. It generates 1000 pair images for training and testing respectively. It trains each models for 11 trials and 25 epochs in each trial.  


  
## Getting Started
- Run `test.py` to train and test the four models. Mean accuracy and its standard deviation for training and testing will print at the end of the 3 cases: baseline structure, structure with weight sharing, and strucutre with auxiliary loss for each model.

- Cross validation: If you want to test cross validation to check learning rate for four models, please set the boolean flag - 'run_cross_validation' variable to True. 
