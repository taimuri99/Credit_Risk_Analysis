# Credit_Risk_Analysis
Module 17 Jupyter Notebooks: Supervised Machine Learning and Credit Risk
## Overview of the loan prediction risk analysis
For this module we used machine learning models to predict outcomes to credit risk. Predicting Credit Risk is important to see likelihood of people repaying Credit loans. This likelihood affects the outcome of banks accepting to give loans to individuals. Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, we employ different techniques to train and evaluate models with unbalanced classes. We use imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling.

Using the credit card credit dataset from LendingClub, we oversample the data using the RandomOverSampler and SMOTE algorithms, undersample the data using the ClusterCentroids algorithm and use a combination approach of oversampling and undersampling using the SMOTEENN algorithm. After that we compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. At the end, we evaluate the performance of these models and make a written recommendation on whether they should be used to predict credit risk. 

## Results
Here are the results of the different machine learning models used and the accuracy scores, confusion matrices and classification reports. The data was split into training and testing sets to first train the models and then use the tested data to predict outcomes.
### Oversampling
#### Naive Random Oversampling

<img width="1095" alt="RandomOversampling" src="https://user-images.githubusercontent.com/87828174/147393708-da61d2bf-60c1-43ab-8280-242f75aad42a.png">

- Balanced Accuracy Score: 0.640324421824783
- Precision: 0.99
- Recall: 0.62
- For High_Risk, precision is 1% while recall is 66%.
- Low_risk which are higher in number have a much better precision of 100% with a recall of 62%.

#### SMOTE Oversampling

<img width="1087" alt="SmoteOversampling" src="https://user-images.githubusercontent.com/87828174/147393797-7af96b95-9285-4766-ade0-9a1f02dbb5d5.png">

- Balanced Accuracy Score: 0.6514992150524688
- Precision: 0.99
- Recall: 0.69
- For High_Risk, precision is 1% while recall is 61%.
- Low_risk which are higher in number have a much better precision of 100% with a recall of 69%.

### Undersampling
#### Cluster Centroids

<img width="1089" alt="Undersampling" src="https://user-images.githubusercontent.com/87828174/147393862-641e09d9-e7b9-45b7-bd96-a7730fd34187.png">

- Balanced Accuracy Score: 0.5447339051023905
- Precision: 0.99
- Recall: 0.40
- For High_Risk, precision is 1% while recall is 69%.
- Low_risk which are higher in number have a much better precision of 100% with a recall of 40%.

### Combination (Over and Under) Sampling

<img width="1091" alt="Combination" src="https://user-images.githubusercontent.com/87828174/147393887-d89188d7-e425-4899-a4bb-7c01de32f8f5.png">

- Balanced Accuracy Score: 0.6550612907408608
- Precision: 0.99
- Recall: 0.56
- For High_Risk, precision is 1% while recall is 75%.
- Low_risk which are higher in number have a much better precision of 100% with a recall of 56%.

### Ensemble Learners
#### Balanced Random Forest Classifier

<img width="994" alt="Random Forests" src="https://user-images.githubusercontent.com/87828174/147393989-7386a805-f6fc-47a3-bf1e-3851e7172268.png">

- Balanced Accuracy Score: 0.6830221521918328
- Precision: 1.00
- Recall: 1.00
- For High_Risk, precision is 88% while recall is 37%.
- Low_risk which are higher in number have a precision of 100% with a recall of 100%.

##### List of features ranked by importance

<img width="534" alt="Rankings" src="https://user-images.githubusercontent.com/87828174/147394027-0ddbc1d6-5050-471e-9dd0-3abfb6aaf7a4.png">

#### Easy Ensemble AdaBoost Classifier

<img width="1055" alt="easyensemble" src="https://user-images.githubusercontent.com/87828174/147394038-34200aac-6ad7-4f4e-9931-6482c7ebafe4.png">

- Balanced Accuracy Score: 0.931601605553446
- Precision: 0.99
- Recall: 0.94
- For High_Risk, precision is 9% while recall is 92%.
- Low_risk which are higher in number have a precision of 100% with a recall of 94%.

## Summary
### Summary of the results
### Which model should be used?
