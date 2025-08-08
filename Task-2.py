import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Titanic-Dataset.csv")

print("Summary Statistics:")
summary_stats = df.describe()
print(summary_stats)

print("\nMedian values:")
median_values = df.median(numeric_only=True)
print(median_values)

print("\nSkewness (absolute > 0.5 is significant):")
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
print(df[numeric_features].skew())

print("\nMissing Values Count:")
print(df.isnull().sum())

numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
print(f"\nNumeric features to visualize: {list(numeric_features)}")

for feature in numeric_features:
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    sns.histplot(df[feature].dropna(), kde=True, color='skyblue')
    plt.title(f'Histogram of {feature}')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[feature], color='lightgreen')
    plt.title(f'Boxplot of {feature}')
    
    plt.tight_layout()
    plt.show()

print("Generating pairplot...")
sns.pairplot(df[numeric_features].dropna())
plt.suptitle("Pairplot of Numeric Features", y=1.02)
plt.show()

print("Correlation Matrix Heatmap:")
correlation_matrix = df[numeric_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix (|r| > 0.5 suggests multicollinearity)')
plt.show()

categorical_features = ['Survived', 'Pclass', 'Sex', 'Embarked']
print("\nCategorical Feature Analysis:")

for feature in categorical_features:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=feature, data=df, palette='viridis')
    plt.title(f'Count of Passengers by {feature}')
    plt.show()
print("""
Key Inferences:
1. Numeric Features:
    'Age' is right-skewed (skew = {:.2f}) and has missing values.
    'Fare' has extreme outliers (high fares).
    'Pclass' and 'Fare' are negatively correlated (r = {:.2f}), suggesting higher classes paid more.

2. Categorical Features:
    Most passengers were male (64%).
    Survival rate was higher for females and 1st-class passengers.
    'Embarked' has missing values (S = Southampton was most common).

3. Next Steps:
    Impute missing 'Age' values (median or predictive model).
    Log-transform 'Fare' to handle skewness.
    One-hot encode categorical features for modeling.
""".format(
    df['Age'].skew(),
    correlation_matrix.loc['Pclass', 'Fare']
))
