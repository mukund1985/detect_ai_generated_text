import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the preprocessed features and target variable
features_path = 'data/processed/preprocessed_train_features.csv'
target_path = 'data/processed/preprocessed_train_target.csv'

X = pd.read_csv(features_path)
y = pd.read_csv(target_path)

# Ensure 'y' is correctly aligned as a Series for model training
y = y['generated']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
