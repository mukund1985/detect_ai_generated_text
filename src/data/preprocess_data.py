# Data Preprocessing

import pandas as pd

# Load the enhanced dataset
data_path = 'data/processed/enhanced_train_essays.csv'
data = pd.read_csv(data_path)

# No need to explicitly add 'generated' column since it's already present
# Perform any required preprocessing steps here
# For example, text cleaning, encoding, etc.

# For the sake of example, let's assume preprocessing is done and we're ready to save
# Save the features (without the 'generated' column) and the target variable separately

# Features - assuming preprocessing has been done and is stored in `data` variable
features = data.drop('generated', axis=1)
preprocessed_features_path = 'data/processed/preprocessed_train_features.csv'
features.to_csv(preprocessed_features_path, index=False)

# Target variable
target = data['generated']
preprocessed_target_path = 'data/processed/preprocessed_train_target.csv'
target.to_csv(preprocessed_target_path, index=False)

print("Preprocessing complete. Features and target variable saved separately.")
