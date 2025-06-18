# Titanic-Survival-Prediction

# Objective:

To build a classification model that predicts whether a passenger survived the Titanic shipwreck based on available personal and voyage details using Logistic Regression.

# Dataset Overview:

Name: Titanic Dataset

Source: Seaborn library (sns.load_dataset('titanic')) or Kaggle Titanic Dataset

Type: Structured dataset

Target Variable: survived (0 = No, 1 = Yes)

# Machine Learning Task:

Type: Supervised Learning

Model: Classification

Algorithm: Logistic Regression

Goal: Predict survived (1 or 0)

# Data Preprocessing Steps:

Remove irrelevant or high-missing columns (deck, embark_town, alive)

Drop rows with missing values

Label Encoding for categorical features like sex, embarked, class, etc.

Split dataset into training and test sets (80/20)

# Model Training & Evaluation:

Model Used: LogisticRegression(max_iter=200) from sklearn

Metrics Evaluated:

Accuracy Score

Classification Report (Precision, Recall, F1-Score)

Confusion Matrix Visualization

# Conclusion:

The Logistic Regression model can reasonably predict Titanic passenger survival using demographic and travel features. It’s a good baseline classification model that can be further improved with techniques like:

Feature engineering

Hyperparameter tuning

Advanced models like Random Forest or XGBoost
