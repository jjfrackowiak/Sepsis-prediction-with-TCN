### Sepsis-prediction-with-TCN

Project based on physionet challenge 2019 - sepsis prediction in hours after admission to ICU unit. The utilised model is a pytorch implementation of Temporal Convulutional Network (TCN). 
The economic goal in this task is to perform diagnosis as early as possible without sacrificing hospital resources for treating falsely dagnosed cases.

### Data processing:

Target (sepsis diagnosis) is already shifted 6 hours back for every patient (patient=sample=one psv file) allowing for teaching the network to recognise the illness earlier than clinician.


#### Training directory structure

Model was trained on a dataset of hourly measurements of 40643 patients registered by two ICU units of two hospitals. Data is publicly available at: https://physionet.org/content/challenge-2019/1.0.0/

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
### TCN architecture 

Utilised architecture is a sequence-to-sequence or many-to-many.
The adventage of this approach over the one with one-step forward prediction is the ability to take account of all provided information unlimited to certan window.

### Training results:

| Metric | Training (Epoch 10/10) | Validation (Epoch 10/10) |
| ------------- | ------------- | ------------- |
| Balanced Accuracy | 0.8049 | 0.8048 |
| AUC Score | 0.8049  | 0.8048 |
| Weighted Loss* | 0.2001  | 0.1160 |
| Positive class frequency | 18451/1207498 | 4563/301643 |
\* Because of severe class inbalance nn.BCEWithLogitsLoss() with pos_weight = 10 was utilised

#### Loss throughout epochs for training (gray) and validation (blue):

<img width="1142" alt="Loss" src="https://github.com/jjfrackowiak/Sepsis-prediction-with-TCN/assets/84077365/aafb9cc5-26f8-49de-a219-f6bdbdb3a4e4">


#### ROC_AUC score throughout epochs for training (gray) and validation (blue):

<img width="1143" alt="Roc_auc" src="https://github.com/jjfrackowiak/Sepsis-prediction-with-TCN/assets/84077365/503c4766-8eea-4cfd-9e08-e9281a0744d2">




