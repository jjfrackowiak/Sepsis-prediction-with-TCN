### Sepsis-prediction-with-TCN
Project based on physionet challenge 2019 - sepsis prediction in hours after admission to ICU unit. The utilised model is a pytorch implementation of TCN.

### Model Training results:

Epoch [12/12], Balanced Accuracy: 0.6632452759551555, Loss: 0.05427886923493302, No. of Sepsis diagnoses out of all obsesrvations : 11357/616289
Mean Epoch Balanced Accuracy: 0.6632452759551555, Mean Epoch Loss: 0.05427886923493302

Loss throught epochs (smoothened and raw):

<img width="1145" alt="Screenshot 2023-10-03 at 09 47 48" src="https://github.com/jjfrackowiak/Sepsis-prediction-with-TCN/assets/84077365/e5d2cd54-752b-4f5b-b797-618458b24bd2">

Balanced accuracy throught epochs (smoothened and raw):

<img width="1145" alt="Screenshot 2023-10-03 at 09 47 34" src="https://github.com/jjfrackowiak/Sepsis-prediction-with-TCN/assets/84077365/8b97028c-a0ff-40de-b709-a174c542c86c">

### Test on a hold-out dataset:

Test Balanced Accuracy: 0.6777955079716917, Test Mean Loss: 0.054629570946417516, No. of Sepsis cases out of all observations : 2782/152912
