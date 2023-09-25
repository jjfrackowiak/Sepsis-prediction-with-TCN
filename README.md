# Sepsis-prediction-with-TCN
Project based on physionet challenge 2019 - sepsis prediction in hours after admission to ICU unit. The utilised model is a pytorch implementation of TCN.

# Model Training results:

Epoch [12/12], Balanced Accuracy: 0.6632452759551555, Loss: 0.05427886923493302, No. of Sepsis diagnoses out of all obs.: 11357/616289
Mean Epoch Balanced Accuracy: 0.6632452759551555, Mean Epoch Loss: 0.05427886923493302

Loss throught epochs (smoothened and raw):

<img width="1142" alt="Screenshot 2023-09-25 at 09 42 07" src="https://github.com/jjfrackowiak/Sepsis-prediction-with-TCN/assets/84077365/71f82571-d94f-41a2-9a2b-64bb09827f48">

Balanced accuracy throught epochs (smoothened and raw):

<img width="1109" alt="Screenshot 2023-09-25 at 13 02 33" src="https://github.com/jjfrackowiak/Sepsis-prediction-with-TCN/assets/84077365/85221358-ad07-48d2-8021-3e21d3ce153c">


# Test on a hold-out dataset:

Confusion Matrix:
  Actual\Predicted   |    0    |    1   |
_____________________|_________|________|
          0          |  149712 |   418  |
_____________________|_________|________|
          1          |  1785   |   997  |
_____________________|_________|________|
Test Balanced Accuracy: 0.6777955079716917, Test Mean Loss: 0.054629570946417516, No. of Sepsis diagnoses out of all obs.: 2782/152912
