# 🚀 Stage 3: Modelling of the STIGMA Project

This is the third stage of the STIGMA project, focusing on the modelling of the STIGMA dataset. The main goal of this stage is to build and evaluate models that can predict the target variable based on the preprocessed data. 🎯

The [`Stage_3_Modelling_STIGMA.ipynb`](command:_github.copilot.openRelativePath?%5B%22Stage_3_Modelling_STIGMA.ipynb%22%5D "Stage_3_Modelling_STIGMA.ipynb") notebook contains the modelling steps, where we build, train, and evaluate various machine learning models. This stage is crucial for finding the best model that can accurately predict the target variable. 📊

Key steps in this last stage include:

## 1. **Models**: 🏗️<br>
Due to the fact that a significant portion of our categorical data is ordinal, we have opted to implement a tree-based model, including the following: <br>
- **Decision Tree 🌳**
- **Random Forest 🌳🌳🌳**
- **AdaBoost🌳👾🌳**
- **XGBoost 🌳🤖🌳**
- **GradientBoosting 🌳🎄🌳**<br> 

## 2. **Model Training and Evaluation**: 🏋️‍♀️🎯<br>
Several of the models that we have identified as potential uses are subsequently trained using the available train data.
<br>

In this modeling, the evaluation metrics that are prioritized are AUC and Recall. Recall is a critical metric in our pursuit to minimize False Negative model predictions, which result in financial losses for the company due to training expenses. AUC, on the other hand, signifies the model's ability to distinguish between positive and negative labels effectively; thus, we aim to identify the model that achieves the highest stable recall and test scores for every iteration.

### Model Results
| Model Name | Recall Train | Recall Test | AUC Train | AUC Test |
|------------|--------------|-------------|-----------|----------|
| Decision Tree | 82 | 66 | 87 | 70 |
| Random Forest | 83 | 67 | 87 | 73 |
| AdaBoost | 70 | 71 | 76 | 77 |
| XGBoost | 81 | 72 | 84 | 76 | 
| GradientBoosting | 80 | 79 | 79 | 78 |

We discovered that the GradientBoosting Model with the highest recall test (79%) and train (80%), stability compared to other models. RandomizedSearch CV and Learning Curve analysis are utilized to determine the optimal hyperparameters for the GradientBoosting model in order to enhance the model's performance :

## 4. **Model Selection**: 🥇<br>

### Model Results
| Model Name | Recall Train | Recall Test | AUC Train | AUC Test |
|------------|--------------|-------------|-----------|----------|
| GradientBoosting | **80** | **79** | **79** | **78** |

We discovered that the GradientBoosting Model with the highest recall test (79%) and train (80%), stability compared to other models. RandomizedSearch CV and Learning Curve analysis are utilized to determine the optimal hyperparameters for the GradientBoosting model in order to enhance the model's performance :
### Best Hyperparameters

| Parameter | Value |
|-----------|-------|
| learning_rate | 0.018571428571428572 |
| n_estimators | 50 |
| min_samples_split | 3 |
| min_samples_leaf | 5 |
| max_leaf_nodes | 9 |
| max_depth | 3 |
| n_iter_no_change | 2 |
| random_state | 17 |
| warm_start | False |


## 5. **Model After Hyperparameter Tuning**: 🚀

The improved recall of our GradientBoosting Model as a result of hyperparameter tuning corresponds to a reduction in the number of incorrectly predicted candidates and training cost loss.

| Model Name | Recall Train | Recall Test | AUC Train | AUC Test |
|------------|--------------|-------------|-----------|----------|
| GradientBoosting | **84** | **85** | **76** | **77** |


This stage is a part of the Rakamin Academy's FinPro program. 🎓