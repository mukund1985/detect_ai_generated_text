import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the preprocessed features and target variable from separate files
features_path = 'data/processed/preprocessed_train_features.csv'
target_path = 'data/processed/preprocessed_train_target.csv'

X = pd.read_csv(features_path)
y = pd.read_csv(target_path)['generated']  # Adjust column name as necessary

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Random Forest classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Model evaluation
predictions = rf_model.predict(X_test)
print("Random Forest Classification Report:")
print(classification_report(y_test, predictions))
