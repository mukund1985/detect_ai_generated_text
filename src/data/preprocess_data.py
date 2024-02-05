import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data_path = 'data/processed/enhanced_train_essays.csv'
data = pd.read_csv(data_path)

# Identify categorical columns and convert them to numeric
# Assuming 'category_column' is a placeholder for actual categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns.drop('generated')
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# Optionally, drop any non-relevant columns (e.g., ID columns)
# data = data.drop(['id_column'], axis=1)

# Separate the target variable and save it separately
target = data['generated']
features = data.drop('generated', axis=1)

# Save the preprocessed data
features.to_csv('data/processed/preprocessed_train_features.csv', index=False)
target.to_csv('data/processed/preprocessed_train_target.csv', index=False)
