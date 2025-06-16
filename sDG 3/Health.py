import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 2. Load datasets
# Download from Kaggle and upload files to Colab, or load from URL if available
# Dataset files: 'dengue_features_train.csv' and 'dengue_labels_train.csv'
features = pd.read_csv('dengue_features_train.csv')
labels = pd.read_csv('dengue_labels_train.csv')

# 3. Explore data
print(features.head())
print(labels.head())

# Merge datasets on common columns
df = features.merge(labels, on=['city', 'year', 'weekofyear'])

# 4. Check for missing values
print(df.isnull().sum())

# Fill missing values (simple approach)
df.fillna(df.mean(), inplace=True)  # Fill numeric columns with mean
# Alternatively, you can use forward fill for time series data  
# df.fillna(method='ffill', inplace=True)  # Forward fill for time series continuity
# For categorical columns, you might want to fill with mode or a specific value
# df['column_name'].fillna(df['column_name'].mode()[0], inplace=True)  # Example for categorical columns    
# Forward fill for time series continuity
df.fillna(method='ffill', inplace=True)
# Alternatively, you can drop rows with missing values
# df.dropna(inplace=True)  # Drop rows with any missing values  
# 5. Visualize data
sns.pairplot(df, hue='city', vars=['total_cases', 'ndvi_ne', 'ndvi_nw', 'precipitation_amt_mm'])

# 5. Define features and target
X = df.drop(columns=['total_cases', 'city', 'year'])  # Drop non-numeric or target columns
y = df['total_cases']

# 6. Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 8. Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 9. Predict and evaluate
y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("R2 score:", r2_score(y_test, y_pred))

# 10. Plot true vs predicted cases
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Dengue Cases")
plt.ylabel("Predicted Dengue Cases")
plt.title("Actual vs Predicted Dengue Cases")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.show()

# 11. Feature importance
feat_importances = pd.Series(model.feature_importances_, index=df.drop(columns=['total_cases', 'city', 'year']).columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Feature Importances')
plt.show()