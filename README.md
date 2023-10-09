### Sepsis-prediction-with-TCN

Project based on physionet challenge 2019 - sepsis prediction in hours after admission to ICU unit. The utilised model is a pytorch implementation of TCN. 

### Data processing:

Target (sepsis diagnosis) is already shifted 6 hours for every case


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
### TCN architecture (sequence-to-sequence model, many-to-many)


### Model Training results:

Epoch [10/10], Balanced Accuracy (Train): 0.8049861799025777, AUC Score (Train): 0.8049861799025778, Loss: 0.20019800427991882, No. of Sepsis diagnoses out of all obs.: 18451/1207498

Epoch [10/10], Balanced Accuracy (Validation): 0.8048171351568002, AUC Score (Validation): 0.8048171351568002, Loss (Validation): 0.11600673198699951, No. of Sepsis diagnoses out of all obs. (Validation): 4563/301643

Loss throughout epochs for training (gray) and validation (blue):

<img width="1142" alt="Loss" src="https://github.com/jjfrackowiak/Sepsis-prediction-with-TCN/assets/84077365/aafb9cc5-26f8-49de-a219-f6bdbdb3a4e4">


ROC_AUC score throughout epochs for training (gray) and validation (blue):

<img width="1143" alt="Roc_auc" src="https://github.com/jjfrackowiak/Sepsis-prediction-with-TCN/assets/84077365/503c4766-8eea-4cfd-9e08-e9281a0744d2">




