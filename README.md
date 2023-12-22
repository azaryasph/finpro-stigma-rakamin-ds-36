# 🔎Stage 1: Exploratory Data Analysis (EDA) of the STIGMA Project

<br>
<p align="center">
<img src="https://user-images.githubusercontent.com/74038190/212749726-d36b8253-74bb-4509-870d-e29ed3b8ff4a.gif" width="500">
</p>


This is the first stage of the STIGMA project, focusing on exploratory data analysis (EDA) of the STIGMA dataset. The main goal of this stage is to understand the data, find patterns, spot anomalies, test hypotheses, and check assumptions with the help of summary statistics and graphical representations.

The [`Stage_1_EDA_STIGMA.ipynb`](command:_github.copilot.openRelativePath?%5B%22Stage_1_EDA_STIGMA.ipynb%22%5D "Stage_1_EDA_STIGMA.ipynb") notebook contains the initial data exploration, where we examine the basic metrics, distributions, and relationships between variables. This stage is crucial for familiarizing ourselves with the data and gaining insights that will help us decide how to handle preprocessing and modeling in the next stages.

## 📚 Installation

This project requires Python and the following Python libraries installed:

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scipy

If you don't have Python installed yet, it's recommended that you install [the Anaconda distribution](https://www.anaconda.com/distribution/) of Python, which already has the above packages and more included.

To install the Python libraries, you can use pip:

```bash
pip install numpy pandas matplotlib seaborn scipy
```
To run the Jupyter notebook, you also need to have Jupyter installed. If you installed Python using Anaconda, you already have Jupyter installed. If not, you can install it using pip:
```bash
pip install jupyter
```

Once you have Python and the necessary libraries, you can run the project using Jupyter Notebook:
```bash
jupyter notebook Stage_1_EDA_STIGMA.ipynb
```
## 📝 Usage/Examples

Provide examples on how to use your project. For example, you can show how to run your `Stage_1_EDA_STIGMA.ipynb` notebook:

```bash
jupyter notebook Stage_1_EDA_STIGMA.ipynb
```

## 📊 About Dataset 
### Dataset consist :
- 19,158 rows and 14 columns
- The target column in this dataset is "target"
- Dataset contains missing values
- The dataset does not have duplicate values

### 📌 Features

| Feature | Explanation |
|---------|-------------|
| enrollee_id | Unique ID for candidate |
| city | City code |
| city_development_index | Development index of the city (scaled) |
| gender | Gender of candidate |
| relevent_experience | Relevant experience of candidate |
| enrolled_university | Type of University course enrolled if any |
| education_level | Education level of candidate |
| major_discipline | Education major discipline of candidate |
| experience | Candidate total experience in years |
| company_size | No of employees in current employer's company |
| company_type | Type of current employer |
| last_new_job | Difference in years between previous job and current job |
| training_hours | Training hours completed |
| target | 0 – Not looking for job change, 1 – Looking for a job change |

### Missing Values Columns

| Column Name | Missing Values Percentage |
|-------------|---------------------------|
| Company Type | 32.05% |
| Company Size | 30.99% |
| Gender | 23.53% |
| Major Discipline | 14.68% |
| Education Level | 2.40% |
| Last Job Tenure | 2.21% |
| Enrolled Status | 2.01% |
| Experience | 0.34% |

<br>
<br>

<p align = "center"> 
    <img src="./assets/img/EDA 3.jpg" width="1000">
</p>

<p align = "center"> 
    <img src="./assets/img/EDA.png" width="1000">
</p>

<p align = "center"> 
    <img src="./assets/img/EDA 2.jpg" width="1000">
</p>

<br>
<br>

 Several significant variables have yielded insights into the attributes of candidates who tend to depart after training and those who continue to be employed by the organization after training. These variables include the candidate's location, status as an employee with relevant data science experience, educational attainment, and work experience.

🎓 This stage is a part of the Rakamin Academy's FinPro program.