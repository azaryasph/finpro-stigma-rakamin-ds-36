# 🚀 Stage 2: Data Preprocessing of the STIGMA Project

<br>
<p align="center">
<img src="https://user-images.githubusercontent.com/74038190/221352995-5ac18bdf-1a19-4f99-bbb6-77559b220470.gif" width="400">
</p>


This is the second stage of the STIGMA project, focusing on data preprocessing of the STIGMA dataset. The main goal of this stage is to clean and transform the raw data to make it suitable for data analysis and modeling. 🧹🔄

The [`Stage_2_Preprocessing_STIGMA.ipynb`](command:_github.copilot.openRelativePath?%5B%22Stage_2_Preprocessing_STIGMA.ipynb%22%5D "Stage_2_Preprocessing_STIGMA.ipynb") notebook contains the preprocessing steps, where we handle missing values, outliers, and categorical variables. This stage is crucial for improving the quality of the data and ensuring that the dataset is a correct and useful representation of the problem to be modeled. 📊

## 📚 Installation

This project requires Python and the following Python libraries installed:

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scipy
- sklearn
- imblearn

If you don't have Python installed yet, it's recommended that you install [the Anaconda distribution](https://www.anaconda.com/distribution/) of Python, which already has the above packages and more included.

To install the Python libraries, you can use pip:

```bash
pip install numpy pandas matplotlib seaborn scipy sklearn imblearn
```
To run the Jupyter notebook, you also need to have Jupyter installed. If you installed Python using Anaconda, you already have Jupyter installed. If not, you can install it using pip:

```bash
pip install jupyter
```

Once you have Python and the necessary libraries, you can run the project using Jupyter Notebook:
```bash
jupyter notebook Stage_2_Preprocessing_STIGMA.ipynb
```
## 📝 Usage/Examples

Provide examples on how to use your project. For example, you can show how to run your `Stage_2_Preprocessing_STIGMA.ipynb` notebook:

```bash
jupyter notebook Stage_2_Preprocessing_STIGMA.ipynb
```
---
<br>

**Key steps in this stage include:**

## 1. **Missing Values Handling** 🕵️‍♀️<br>
The imputation method is used to populate the missing values; the median frequency of each feature with missing values is utilized to impute values to those features. The efficacy of this median frequency filling method in enhancing our machine learning classification model was validated through a series of iterative experiments involving different imputing techniques.

## 2. **Feature Transformation** 🔄<br>
In order to enhance the future performance of the machine learning model and account for the large number of unique values among the categorical features in this dataset, a feature transformation is implemented on the feature values to make them more general. Indeed, in order to package the information rather than drastically reduce it, we have conducted research on this transformation (generalizing) feature across multiple articles.

## 3. **Feature Selection** 🎯<br>
Using the chi-square test, we examine the correlation between the target and the categorical features (strings) at hand. Once we have determined which features are correlated with the target via the features, we eliminate those that are less correlated with the target.

## 4. **Feature Encoding** 🏷️<br>
We encode all of our categorical features (strings) using the label encoding method, given that our features are ordinal data and the majority of machine learning algorithms perform better with numerical data.

## 5. **Handle Imbalance Data** ⚖️<br>
Before handling imbalanced data, the dataset was divided into two subsets: test (30%) and train (70%) sets.
The imbalance data class consists of 14,381 candidates, or 75.1% of the total, who are not seeking a job change, as opposed to 4,777 candidates, or 24.9%, who are seeking a job change. SMOTE is employed to rectify this disparity by oversampling the minority class.
<br>
<br>
<br>
### The data is clean and prepared for machine learning to predict the target after undergoing all of this preprocessing.🤖

This stage is a part of the Rakamin Academy's FinPro program. 🎓