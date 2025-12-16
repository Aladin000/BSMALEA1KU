# IT-Universität Kopenhagen - Tryg - Machine Learning

## Overview

This project explores the problem of predicting claims risk in automobile insurance. The goal is to estimate the expected claim risk for each insurance policy based on vehicle and driver characteristics (ITU, 2025).

## Methods

Three different modeling approaches are used:

- A **Decision Tree Regressor**, implemented from scratch  
- A **Feed-Forward Neural Network**, implemented from scratch  
- A **Random Forest**, implemented using *scikit-learn* (Scikit-learn, 2025)

For comparison and validation, *scikit-learn* reference implementations are also used for the custom-built models.

## Evaluation Strategy

All models are evaluated using **5-fold cross-validation** during hyperparameter tuning. A separate **held-out test set** is reserved exclusively for final performance evaluation to ensure unbiased results.

## Project Scope

The accompanying report covers:
- Data preparation and preprocessing steps  
- Implementation details of each modeling approach  
- A comparative analysis of model performance  
