import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
file_path = "dataset/kepler_data_subset.csv"
df = pd.read_csv(file_path)

# Drop the 'kepler_name' column as it's not useful for modeling
df.drop('kepler_name', axis=1, inplace=True)

# Check for and drop duplicates
df.drop_duplicates(inplace=True)

# Directly selecting numerical and categorical columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_columns = df.select_dtypes(exclude=['int64', 'float64']).columns.tolist()

# Impute missing values for numerical columns
imputer = SimpleImputer(strategy='mean')
df[numerical_columns] = imputer.fit_transform(df[numerical_columns])

# Normalize numerical columns
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Label encode categorical columns, if any
label_encoder = LabelEncoder()

# Apply label encoding to each categorical column
for col in categorical_columns:
    if df[col].dtype == 'object':  # Check if column contains non-numeric data
        df[col] = label_encoder.fit_transform(df[col])

# Label encode the 'koi_disposition' column before dropping it
df['koi_disposition_encoded'] = label_encoder.fit_transform(df['koi_disposition'])
df.drop(columns=['koi_disposition'], inplace=True)

# Check if all columns are numeric now
print(f"Columns after preprocessing: {df.columns.tolist()}")

# Heatmap of the correlation matrix
plt.figure(figsize=(12, 8))
correlation_matrix = df[numerical_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Split the data into training and testing sets (80% train, 20% test)
X = df.drop(columns=['koi_disposition_encoded'])
y = df['koi_disposition_encoded']

# Ensure all features are numeric
if X.select_dtypes(exclude=['int64', 'float64']).shape[1] > 0:
    print("Warning: There are non-numeric columns in the feature matrix X. Converting them.")
    X = pd.get_dummies(X)  # One-hot encode any categorical features if necessary

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Logistic Regression model
log_reg = LogisticRegression(max_iter=1000)

# Train the model
log_reg.fit(X_train, y_train)

# Make predictions
y_pred = log_reg.predict(X_test)

# Visualizing the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification Report
class_report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(class_report)

# Optionally, print the coefficients of the logistic regression model
coefficients = pd.DataFrame(log_reg.coef_, columns=X.columns)
print("\nLogistic Regression Coefficients:")
print(coefficients)
