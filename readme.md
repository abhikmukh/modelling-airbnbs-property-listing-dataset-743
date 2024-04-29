# Modelling of airBNB property listing dataset
## Overview
This project is a part of the data science project to predict the price of airBNB property listing. The initial target variable is price_per_night and the rest of the columns are the features. The dataset is cleaned and preprocessed before modelling. The modelling is done using classification, regression and neural network models. The best model is selected based on the performance metrics. The hyperparameter tuning is done using ray tune. The best model is then used to predict the price of the airBNB property listing. After that the frame work has been used to predict a different target "bedrooms".

## Details of work done
- Processed and cleaned dataset using Pandas to improve data quality.
- Visually analysed the dataset to understand it better. 
- Trained, compared and evaluated machine learning models (Random Forest,
Linear/Logistic Regression, XGBoost etc) for classification (determining different
Airbnb categories) & regression (predicting tariff) use cases.
● Performed hyperparameter tuning and cross validation using GridSearchCV for ML to optimise the results for
particular metrics, such as precision in classification and ray tune for Pytorch.
- Used Tensorboard to visualise the hyperparameter tuning results.
## How to run
modelling.py calls three scripts classification.py, regression.py and hyper_nn.py and runs the whole pipeline.
modelling_utils.py contains all the modelling related functions used in the scripts and eda_utils.py has other util functions. tabular_data.py does the cleaning
reuse_pipeline.py is used to predict the target "bedrooms" using the automated framework created in the project.
```bash

## tensorboard images of hyper parameter tuning
![hyper_nn](screenshots/ray_tune1.PNG)
![hyper_nn](screenshots/tf1.PNG)
![hyper_nn](screenshots/tf2.PNG)

```bash