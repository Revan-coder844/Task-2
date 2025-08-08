# Task-2
Titanic Dataset Exploratory Data Analysis (EDA)
# Overview
This repository, task-2, contains the code and resources for performing Exploratory Data Analysis (EDA) on the Titanic dataset as part of the AI & ML Internship Task 2. The goal is to understand the dataset using statistical summaries and visualizations, identify patterns, trends, and anomalies, and derive feature-level inferences to inform further modeling.
# Dataset
The dataset used is the Titanic Dataset (Titanic-Dataset.csv), which contains information about passengers on the Titanic, including features like PassengerId, Survived, Pclass, Sex, Age, Fare, Embarked, and more.
 # Tools Used

Pandas: For data manipulation and summary statistics.
Matplotlib: For creating static visualizations.
Seaborn: For enhanced visualizations like histograms, boxplots, pairplots, and correlation heatmaps.
Plotly: For interactive visualizations, such as scatter plots.

# Analysis Performed
The EDA process followed the steps outlined in the task:

# Summary Statistics:

Generated descriptive statistics (mean, median, std, etc.) for numeric features using df.describe().
Calculated median values separately for numeric features.
Identified missing values in the dataset (e.g., Age has 177 missing values, Cabin has 687 missing values).
Quantified skewness for numeric features using df.skew() to confirm distributional properties.


# Histograms and Boxplots:

Created histograms (with KDE) and boxplots for numeric features (PassengerId, Survived, Pclass, Age, SibSp, Parch, Fare) to visualize distributions and detect outliers.
Observations:
Age is right-skewed with some missing values.
Fare has significant outliers (high fare values), with IQR-based detection identifying 116 outliers.
SibSp and Parch show most passengers had small family sizes.




# Categorical Feature Analysis:

Visualized survival counts and rates for categorical features (Sex, Pclass, Embarked) using count plots and bar plots.
Observations:
Females had a significantly higher survival rate than males.
First-class passengers (Pclass=1) had the highest survival rate.
Passengers embarking from Cherbourg (Embarked=C) showed a higher survival rate.




# Pairplot and Correlation Matrix:

Generated a pairplot to visualize relationships between numeric features.
Created a correlation matrix heatmap to show correlations (e.g., strong negative correlation of -0.55 between Pclass and Fare).


# Interactive Visualization:

Created an interactive Plotly scatter plot of Age vs. Fare, colored by Survived, with hover data for Pclass and Sex to explore relationships interactively.


# Patterns, Trends, and Anomalies:

Identified right skewness in Age (skewness: 0.39) and Fare (skewness: 4.79).
Noted 116 outliers in Fare using IQR-based detection, indicating the need for transformation before modeling.
Observed strong relationships between Pclass and survival rates, with first-class passengers having higher survival rates.
Highlighted missing values in Age (177) and Cabin (687) as areas to address, with a suggestion to impute Age with the median.


# Feature-Level Inferences:

Age: Missing values (177) and moderate right skewness (0.39) suggest imputation with the median and possible log transformation.
Fare: Significant outliers (116) and high skewness (4.79) indicate the need for scaling or log transformation.
Pclass: Strongly influences survival, with first-class passengers having a survival rate of ~63%.
Sex: Females had a survival rate of 74%, much higher than males (19%).
Embarked: Passengers from Cherbourg had a higher survival rate (55%) compared to Southampton (34%) and Queenstown (~39%).
SibSp and Parch: Indicate family size, with potential negative correlation to survival for larger families.
Strong negative correlation between Pclass and Fare (-0.55) suggests wealthier passengers were in higher classes.
View the generated plots (saved in the plots/ directory) and printed outputs in the console.
