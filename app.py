import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import streamlit as st

import numpy as np
import pandas as pd
import seaborn as sns



# Pima Indians Diabetes Analysis
st.title("Pima Indians Diabetes Analysis")

# This analysis was done by: Salah Al-Basha
st.write("This analysis was done by: **Salah Al-Basha**")

# Description
st.write("### Description")
st.write("Diabetes is one of the most frequent diseases worldwide and the number of diabetic patients are growing over the years. The main cause of diabetes remains unknown, yet scientists believe that both genetic factors and environmental lifestyle play a major role in diabetes.")
st.write("A few years ago research was done on a tribe in America which is called the Pima tribe. In this tribe, it was found that the ladies are prone to diabetes very early. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients were females at least 21 years old of Pima tribe.")

# Objective
st.write("### Objective")
st.write("Here, we are analyzing different aspects of Diabetes in the Pima Diabetes Analysis by doing Exploratory Data Analysis.")

# Data Dictionary
st.write("### Data Dictionary")
st.write("The dataset has the following information:")
st.write("- **Pregnancies**: Number of times pregnant")
st.write("- **Glucose**: Plasma glucose concentration over 2 hours in an oral glucose tolerance test")
st.write("- **BloodPressure**: Diastolic blood pressure (mm Hg)")
st.write("- **SkinThickness**: Triceps skin fold thickness (mm)")
st.write("- **Insulin**: 2-Hour serum insulin (mu U/ml)")
st.write("- **BMI**: Body mass index (weight in kg/(height in m)^2)")
st.write("- **DiabetesPedigreeFunction**: A function that scores the likelihood of diabetes based on family history.")
st.write("- **Age**: Age in years")
st.write("- **Outcome**: Class variable (0: a person is not diabetic or 1: a person is diabetic)")

# Read the dataset
pima = pd.read_csv("diabetes.csv")

# Display the head of the dataset
st.write("#### Displaying the first 10 records of the dataset")
st.write(pima.head(10))

# Display the tail of the dataset
st.write("#### Displaying the last 10 records of the dataset")
st.write(pima.tail(10))

# Display the shape of the dataset
st.write("#### Shape of the dataset")
st.write(pima.shape)

# Display the total number of elements in the dataset
st.write("The total number of elements is:", pima.size)

# Display the data types of the variables
st.write("#### Data types of the variables")
st.write(pima.dtypes)
st.write("The BMI and DiabetesPedigreeFunction are floats. The rest of the variables are integers.")

# Display if there are any missing values in the dataset
st.write("#### Missing values")
st.write("Are there any missing values in the dataset?", pima.isnull().values.any())

# Display the summary statistics for all variables except 'Outcome'
st.write("#### Summary statistics for all variables except 'Outcome'")
st.write(pima.iloc[:, 0:8].describe())

"""
I got the transpose of the matrix so that the summary statistics of the data are easier to read.

Let's take a look at what we know about the 'Age' variable:

- There are 768 observations in the 'Age' variable.
- The average age is approximately 33.24 years.
- The standard deviation of the age is approximately 11.76 years.
- The minimum age is 21 years and the maximum age is 81 years.
- 25% of the observations have an age of 24 years or less.
- 50% of the observations have an age of 29 years or less.
- 75% of the observations have an age of 41 years or less.

Therefore, we can conclude that the 'Age' variable has a wide range of values and a relatively high standard deviation, indicating that there is considerable variability in the ages of the individuals in the dataset. Additionally, the median age of 29 years is lower than the mean age of 33.24 years, which suggests that the distribution of ages may be skewed towards older ages.
"""

# Distribution plot for the variable 'BloodPressure'
st.write("#### Distribution plot for Blood Pressure")
fig, ax = plt.subplots()
sns.kdeplot(data=pima, x='BloodPressure', ax=ax)
ax.set_xlabel('Blood Pressure')
ax.set_title('Distribution of Blood Pressure')

# Displaying the figure in Streamlit
st.pyplot(fig)
st.write("The '**BloodPressure**' variable refers to diastolic blood pressure (mm Hg). The histogram displays a peak at 72 (mm Hg) at a density of approximately 0.037. We can also see that it's a normal distribution with low standard deviation. The mean is 72.250000 which is very close to the median of 72; this is why the curve is not skewed.")

# What is the 'BMI' of the person having the highest 'Glucose'?
highest_glucose_bmi = pima[pima['Glucose'] == pima['Glucose'].max()]['BMI']

# What is the mean of the variable 'BMI'?
mean_bmi = pima['BMI'].mean()

# What is the median of the variable 'BMI'?
median_bmi = pima['BMI'].median()

# What is the mode of the variable 'BMI'?
mode_bmi = pima['BMI'].mode()[0]

# Are the three measures of central tendency equal?
st.write("### Are the three measures of central tendency equal?")
st.write("No, the three measures of central tendency are not equal. The mean BMI value is", mean_bmi,
         ", the median BMI value is", median_bmi, ", and the mode BMI value is", mode_bmi,
         ". The mean and median are different, which suggests that there might be some outliers or extreme values in the dataset that are affecting the mean. The fact that the mode is the same as the median suggests that the distribution of BMI values may be somewhat symmetric or roughly bell-shaped.")

# How many women's 'Glucose' levels are above the mean level of 'Glucose'?
above_mean_glucose_count = pima[pima['Glucose'] > pima['Glucose'].mean()].shape[0]
st.write("#### How many women's 'Glucose' levels are above the mean level of 'Glucose'?")
st.write(above_mean_glucose_count)

# How many women have their 'BloodPressure' equal to the median of 'BloodPressure' and their 'BMI' less than the median of 'BMI'?
women_count = pima[(pima['BloodPressure'] == pima['BloodPressure'].median()) & (pima['BMI'] < pima['BMI'].median())].shape[0]
st.write("#### How many women have their 'BloodPressure' equal to the median of 'BloodPressure' and their 'BMI' less than the median of 'BMI'?")
st.write(women_count)

# Creating a pairplot for the variables 'Glucose', 'SkinThickness', and 'DiabetesPedigreeFunction'
st.write("#### Pairplot for Glucose, Skin-thickness and Diabetes Pedigree Function")
pairplot = sns.pairplot(data=pima, vars=['Glucose', 'SkinThickness', 'DiabetesPedigreeFunction'], hue='Outcome')
st.pyplot(pairplot)

"""
#### Observations:

Shown is a scatterplot matrix of the three variables: Glucose, SkinThickness, and DiabetesPedigreeFunction, with different colors for the two groups defined by the Outcome variable (0 for non-diabetic and 1 for diabetic).

By analyzing the plots we can infer the following:


- The analysis is focused on examining the relationships between Glucose, SkinThickness, and DiabetesPedigreeFunction, and how they may be related to the Outcome variable. There is no correlation between any of the variables in the scatterplots. It is important to note, however, that the lack of correlation in a scatterplot does not necessarily mean that there is no relationship between the variables. Therefore, it is always important to carefully examine the data and consider additional statistical analyses to better understand the relationship between three variables.

- The scatterplot matrix shows pairwise scatterplots of the three variables against each other, as well as histograms of each variable along the diagonal. It can be seen in the Glucose diagonal plot, for instance, that non-diabetics have a more right-skewed distribution than diabetics. This suggests that the distribution of Glucose values for non-diabetics is shifted towards higher values compared to diabetics. This observation could potentially have important implications for the identification of patients with diabetes. More specifically, this observation could mean that higher Glucose values are associated with a lower risk of diabetes. In other words, it may be easier for individuals with higher Glucose values to maintain normal blood sugar levels and avoid developing diabetes, compared to those with lower Glucose values.

- Since non-diabetics also had a more right-skewed distribution than diabetics on the SkinThickness and DiabetesPedigreeFunction diagonal plots, this could suggest that there are potential differences in the underlying factors that contribute to the development of diabetes in the Pima population. For example, higher SkinThickness values have been associated with higher levels of insulin resistance, which is a key factor in the development of type 2 diabetes. Therefore, if non-diabetics have higher SkinThickness values than diabetics, this could suggest that non-diabetic individuals in the Pima population may have developed some degree of insulin resistance, but have been able to maintain normal blood sugar levels through other mechanisms, such as lifestyle or genetics.

- Similarly, the DiabetesPedigreeFunction variable is a measure of the likelihood of diabetes based on family history. If non-diabetics have higher DiabetesPedigreeFunction values than diabetics, this could suggest that the family history of diabetes may play a different role in the development of diabetes in the Pima population compared to other populations.

However, it is important to note that this is just a preliminary observation, and further analysis and statistical testing would be necessary to confirm any such relationships.
"""

# Ploting the scatterplot between 'Glucose' and 'Insulin'
st.write("#### Scatterplot between 'Glucose' and 'Insulin'")
fig, ax = plt.subplots()
scatterplot = sns.scatterplot(x='Glucose', y='Insulin', data=pima, ax=ax)
ax = scatterplot
st.pyplot(fig)

"""
#### Observations:

The scatterplot shows a positive relationship between the Glucose and Insulin variables, but there are many data points in a horizontal line. This suggests that there may be a threshold effect or saturation point in the relationship between these two variables.

In other words, as Glucose levels increase, there may be a point beyond which Insulin levels do not increase any further, leading to the horizontal line in the scatterplot. This could be due to the body's ability to produce and utilize insulin, and could be an indication of insulin resistance.
"""

# Ploting the boxplot for the 'Age' variable
st.write("#### Boxplot for 'Age'")
fig, ax = plt.subplots()
boxplot = sns.boxplot(x='Outcome', y='Age', data=pima, ax=ax)
ax = boxplot
st.pyplot(fig)

"""
The outliers in the box plot suggest the presence of individuals who are much older than the typical age range of the population.
"""

# Ploting histograms for the 'Age' variable
no_diabetes_df = pima[pima['Outcome'] == 0]
fig, ax = plt.subplots()
sns.histplot(data=no_diabetes_df, x='Age', bins=5, ax=ax)
ax.set_title('Distribution of Age for Women who do not have Diabetes')
ax.set_xlabel('Age')
ax.set_ylabel('Frequency')
st.pyplot(fig)

"""
#### Observations:
The first histogram shows the distribution of age for women who do not have diabetes, while the second histogram shows the distribution of age for women who have diabetes.

In the first histogram, we can see that the majority of women who do not have diabetes are in the age range of approximately 20-45 years old. There is also a small peak in the age range of approximately 55-60 years old. The histogram is skewed to the right, indicating that there are relatively fewer women in the older age ranges.
"""

# Ploting histograms for the 'Age' variable
fig, ax = plt.subplots()
ax.hist(pima[pima['Outcome'] == 1]['Age'], bins=5)
ax.set_title('Distribution of Age for Women who have Diabetes')
ax.set_xlabel('Age')
ax.set_ylabel('Frequency')
st.pyplot(fig)


"""
#### Observations:
In the second histogram, we can see that the distribution of age for women with diabetes is somewhat similar to the distribution for women without diabetes, with a majority of individuals in the age range of approximately 20-45 years old. However, there is a larger proportion of women with diabetes in the older age ranges, and the histogram is less skewed to the right compared to the histogram for women without diabetes.

Overall, the histograms suggest that age is not a strong predictor of diabetes, as the distributions of age for women with and without diabetes are relatively similar. However, there appears to be a slightly higher prevalence of diabetes among women in the older age ranges.
"""

# Calculate the interquartile range (IQR)
Q1 = pima.quantile(0.25)
Q3 = pima.quantile(0.75)
IQR = Q3 - Q1

# Display the IQR
st.write("#### Interquartile Range (IQR)")
st.write(IQR)

st.write("#### Boxplot of Variables")
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=pima, ax=ax)
plt.xticks(rotation=45, ha='right')

# Display the plot using st.pyplot()
st.pyplot(fig)



"""
The provided code calculates the Interquartile Range (IQR) for each variable in the dataset pima. The IQR is a measure of variability that represents the spread of the middle 50% of the data. It is computed by finding the difference between the third quartile (Q3) and the first quartile (Q1) of the dataset.

The resulting IQR values for each variable in the pima dataset are as follows:

- Pregnancies: 5.0000
- Glucose: 40.5000
- BloodPressure: 16.0000
- SkinThickness: 12.0000
- Insulin: 48.2500
- BMI: 9.1000
- DiabetesPedigreeFunction: 0.3825
- Age: 17.0000
- Outcome: 1.0000
"""

# Ploting the correlation matrix heatmap
st.write("#### Correlation Matrix Heatmap")
correlation_matrix = pima.corr()
fig, ax = plt.subplots()
heatmap = sns.heatmap(correlation_matrix, annot=True, ax=ax)
ax = heatmap
st.pyplot(fig)
"""
#### Observations:

- The diagonal of the matrix is all 1's since each variable is perfectly correlated with itself.
- The highest positive correlation (0.47) is observed between the variables 'Age' and 'Pregnancies'.
- There is also a moderate positive correlation (around 0.3) between 'Glucose' and 'Outcome', 'BMI' and 'SkinThickness', and 'BMI' and 'BloodPressure'.
- There is no significant negative correlation between any of the variables.

## Conclusion:
In this Pima Indians Diabetes Analysis, we explored various aspects of diabetes by conducting exploratory data analysis. The dataset consisted of 768 instances, with 9 variables including Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, and Outcome.

#### Our analysis revealed the following key findings:
1. The dataset had no missing values, ensuring data completeness.
2. The 'Age' variable exhibited a wide range of values with a relatively high standard deviation, indicating significant variability in the ages of individuals in the dataset.
3. The 'BloodPressure' variable followed a normal distribution with a peak at 72 mm Hg and low standard deviation.
4. The mean BMI was 32.45, while the median and mode were 32.0. This suggests a potential presence of outliers or extreme values affecting the mean.
5. There were 343 women with glucose levels above the mean glucose level.
6. Among women with blood pressure equal to the median and BMI less than the median, there were 22 individuals.
7. Pairwise scatterplots showed no significant correlations between Glucose, SkinThickness, and DiabetesPedigreeFunction, but further analysis is required to fully understand their relationships.
8. A scatterplot of Glucose and Insulin revealed a positive relationship, potentially indicating a threshold effect or saturation point beyond which Insulin levels do not increase further.

#### Based on these findings, we offer the following observations and recommendations:
1. Further investigation is needed to understand the factors contributing to the variability in diabetes prevalence among the Pima tribe, considering genetic and environmental factors.
2. In-depth statistical analyses should be conducted to determine the significance of relationships between variables and their impact on diabetes risk.
3. Identifying additional variables and conducting more advanced modeling techniques could improve the accuracy of diabetes prediction and help develop personalized interventions.
4. Collaborating with healthcare professionals and genetic experts may provide valuable insights into the underlying mechanisms of diabetes development within the Pima tribe.

Overall, this analysis provides initial insights into the Pima Indians' diabetes prevalence, highlighting the importance of continued research and targeted interventions to address this significant health issue.
"""
