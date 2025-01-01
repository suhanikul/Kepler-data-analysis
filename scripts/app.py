import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

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

# -------------------------------- Feature Selection for Clustering --------------------------------
# Use only numerical columns except 'koi_disposition_encoded'
X_cluster = df[numerical_columns].drop(columns=['koi_disposition_encoded'], errors='ignore')

# -------------------------------- Elbow Method to Find Optimal K --------------------------------
# Calculate the inertia (within-cluster sum of squares) and silhouette scores
inertia = []
silhouette_scores = []
k_values = range(2, 11)  # Checking clusters from 2 to 10

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_cluster)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_cluster, kmeans.labels_))

# Plot the Elbow Method graph
plt.figure(figsize=(10, 5))
plt.plot(k_values, inertia, 'o-', label='Inertia')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.legend()
plt.show()

# Plot Silhouette Scores
plt.figure(figsize=(10, 5))
plt.plot(k_values, silhouette_scores, 'o-', color='green', label='Silhouette Score')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Scores for Clusters")
plt.legend()
plt.show()

# -------------------------------- K-Means Clustering --------------------------------
# Choose the optimal k (e.g., 3) based on the elbow method
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_cluster)

print(f"Cluster Centers:\n{kmeans.cluster_centers_}")

# -------------------------------- Visualize Clusters using PCA --------------------------------
# Reduce data to 2 dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_cluster)

# Add PCA components and cluster labels to the DataFrame
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

# Scatter plot of clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', palette='Set1', data=df, s=100)
plt.title("K-Means Clusters (PCA Reduced)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.show()

# -------------------------------- Analyze Cluster Results --------------------------------
print("\nCluster Analysis:")
for cluster in range(optimal_k):
    print(f"Cluster {cluster}:")
    print(df[df['Cluster'] == cluster].describe())
