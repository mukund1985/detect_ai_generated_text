# Data Preprocessing 

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the enhanced dataset
data_path = 'data/processed/enhanced_train_essays.csv'
data = pd.read_csv(data_path)

# Add the 'generated' column to the DataFrame
X = data.copy()  # Create a copy of the data to avoid modifying the original
# Assuming 'target_column' is your target variable
X['generated'] = data['generated']

# Print the first few rows to check if 'generated' column is added
print(X.head())

# Identify non-numeric columns for one-hot encoding 
non_numeric_columns = X.select_dtypes(exclude=[np.number]).columns.tolist()

# Drop columns that are non-numeric and not useful for the model (if any)
# Example: X = X.drop(['unnecessary_column'], axis=1)

# Set up preprocessing for categorical data (one-hot encoding)
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), non_numeric_columns)
    ], remainder='passthrough'
)

# Separate features and target variable
y = X['generated']  # Target variable
X = X.drop('generated', axis=1)  # Features

# Preprocess the data using for non-numeric columns using ColumnTransformer
X = preprocessor.fit_transform(X)

# Save the preprocessed data if needed
# For example, you can save it to a new CSV file
preprocessed_data_path = 'data/processed/preprocessed_train_data.csv'
pd.DataFrame(X).to_csv(preprocessed_data_path, index=False)
