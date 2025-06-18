# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load the Titanic dataset from seaborn
df = sns.load_dataset('titanic')

# Display initial information
print("Initial Dataset Shape:", df.shape)
print(df.head())

# Drop irrelevant or high-missing columns
df = df.drop(['deck', 'embark_town', 'alive'], axis=1)

# Drop rows with missing values
df = df.dropna()
print("\nDataset Shape After Dropping NA:", df.shape)

# Encode categorical variables
label_enc = LabelEncoder()
df['sex'] = label_enc.fit_transform(df['sex'])             # male:1, female:0
df['embarked'] = label_enc.fit_transform(df['embarked'])   # C:0, Q:1, S:2
df['class'] = label_enc.fit_transform(df['class'])         # First:0, Second:1, Third:2
df['who'] = label_enc.fit_transform(df['who'])             # man:1, woman:2, child:0
df['adult_male'] = df['adult_male'].astype(int)            # True:1, False:0
df['alone'] = df['alone'].astype(int)                      # True:1, False:0

# Define features and target
X = df.drop('survived', axis=1)
y = df['survived']

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
