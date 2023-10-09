### Sepsis-prediction-with-TCN

Project based on [Physionet challenge 2019](https://moody-challenge.physionet.org/2019/) - sepsis prediction in hours after admission to ICU unit. The utilised model is a PyTorch implementation of Temporal Convulutional Network (TCN). 
The economic goal in this task is to perform diagnosis as early as possible without sacrificing hospital resources for treating falsely identified cases.

#### Training directory structure

Model was trained on a dataset of hourly measurements of 40643 patients registered by two ICU hospital units. Data is publicly available at: [Link](https://physionet.org/content/challenge-2019/1.0.0/)

```python
── training
    ├── index.html
    ├── training_setA
    │   ├── index.html
    │   ├── p000001.psv
    │   ├── p000002.psv
    │   ├── p000003.psv
    │   ├── p000004.psv
    │   ├── p000005.psv
    │   ├── ...
    │   └── p020643.psv
    └── training_setB
        ├── index.html
        ├── p100001.psv
        ├── p100002.psv
        ├── p100003.psv
        ├── p100004.psv
        ├── p100005.psv
        ├── ...
        └── p120000.psv
```

### Data processing:

Target (sepsis diagnosis) is already marked 6 hours before clinical diagnosis (for every patient/sample) allowing for teaching the network to recognise sepsis earlier than clinician (yet not too early). Missing values have been interpolated or median-filled whenever possible. Remaining NaN's have been replaced by zeroes. 

Each sample has been either truncated or NaN-padded to the length of 100 observations, under assumption that sepsis has been diagnosed in the first 100 hours or not diagnosed at all.

### TCN architecture 

Utilised architecture is a sequence-to-sequence or many-to-many Temporal Convolutional Net. 
The advantage of this approach over one-step forward prediction is the ability to take account of all provided information unlimited to selected window.

```python
TCN summary:
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv1d-1              [-1, 150, 42]          45,150
            Conv1d-2              [-1, 150, 42]          45,150
           Chomp1d-3              [-1, 150, 40]               0
           Chomp1d-4              [-1, 150, 40]               0
              ReLU-5              [-1, 150, 40]               0
              ReLU-6              [-1, 150, 40]               0
           Dropout-7              [-1, 150, 40]               0
           Dropout-8              [-1, 150, 40]               0
            Conv1d-9              [-1, 150, 42]          67,650
           Conv1d-10              [-1, 150, 42]          67,650
          Chomp1d-11              [-1, 150, 40]               0
          Chomp1d-12              [-1, 150, 40]               0
             ReLU-13              [-1, 150, 40]               0
             ReLU-14              [-1, 150, 40]               0
          Dropout-15              [-1, 150, 40]               0
          Dropout-16              [-1, 150, 40]               0
           Conv1d-17              [-1, 150, 40]          15,150
             ReLU-18              [-1, 150, 40]               0
    TemporalBlock-19              [-1, 150, 40]               0
           Conv1d-20              [-1, 150, 44]          67,650
           Conv1d-21              [-1, 150, 44]          67,650
          Chomp1d-22              [-1, 150, 40]               0
          Chomp1d-23              [-1, 150, 40]               0
             ReLU-24              [-1, 150, 40]               0
             ReLU-25              [-1, 150, 40]               0
          Dropout-26              [-1, 150, 40]               0
          Dropout-27              [-1, 150, 40]               0
           Conv1d-28              [-1, 150, 44]          67,650
           Conv1d-29              [-1, 150, 44]          67,650
          Chomp1d-30              [-1, 150, 40]               0
          Chomp1d-31              [-1, 150, 40]               0
             ReLU-32              [-1, 150, 40]               0
             ReLU-33              [-1, 150, 40]               0
          Dropout-34              [-1, 150, 40]               0
          Dropout-35              [-1, 150, 40]               0
             ReLU-36              [-1, 150, 40]               0
    TemporalBlock-37              [-1, 150, 40]               0
  TemporalConvNet-38              [-1, 150, 40]               0
           Linear-39                  [-1, 100]          15,100
================================================================
Total params: 526,450
Trainable params: 526,450
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 1.95
Forward/backward pass size (MB): 1.77
Params size (MB): 2.01
Estimated Total Size (MB): 5.73
----------------------------------------------------------------
```

Output of the last fully connected layer is later fed to a loss function of class nn.BCEWithLogitsLoss(), which allows for weightening loss between imbalanced classes.


### Training results:

| Metric | Training (Epoch 10/10) | Validation (Epoch 10/10) |
| ------------- | ------------- | ------------- |
| Balanced Accuracy | 0.8049 | 0.8048 |
| AUC Score | 0.8049  | 0.8048 |
| Weighted Loss* | 0.2001  | 0.1160 |
| Positive class frequency | 18451/1207498 | 4563/301643 |

\* Because of severe class imbalance nn.BCEWithLogitsLoss() with pos_weight = 10 was utilised. Weightening may cause the loss value to be inflated and not directly interpretable.

#### Loss throughout 10 epochs for training (gray) and validation (blue):

<img width="900" alt="Loss" src="https://github.com/jjfrackowiak/Sepsis-prediction-with-TCN/assets/84077365/aafb9cc5-26f8-49de-a219-f6bdbdb3a4e4">


#### ROC_AUC score throughout 10 epochs for training (gray) and validation (blue):

<img width="900" alt="Roc_auc" src="https://github.com/jjfrackowiak/Sepsis-prediction-with-TCN/assets/84077365/503c4766-8eea-4cfd-9e08-e9281a0744d2">




