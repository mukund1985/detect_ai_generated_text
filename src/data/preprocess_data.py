import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data_path = 'data/processed/enhanced_train_essays.csv'
data = pd.read_csv(data_path)

# Debugging: Print columns to verify 'generated' column presence
print("Columns in dataset:", data.columns.tolist())

# Check if 'generated' column exists in the DataFrame
if 'generated' in data.columns:
    # Separate the target variable and features
    target = data['generated']
    features = data.drop('generated', axis=1)
else:
    print("'generated' column not found. Please check the dataset.")
    features = data  # Assuming the rest of the data can still be processed

# Identify categorical columns excluding any ID-like columns if present
# Note: Replace 'id_column' with the actual name of any ID columns you have
categorical_columns = features.select_dtypes(include=['object']).columns.drop(['id_column'], errors='ignore')

# Encode categorical columns
for col in categorical_columns:
    le = LabelEncoder()
    # Handle missing data in categorical columns; fill with a placeholder
    features[col] = features[col].fillna('missing')
    features[col] = le.fit_transform(features[col])

# At this point, `features` should be ready for modeling, and `target` contains the 'generated' labels if present

# Save the processed features and target to new CSV files
preprocessed_features_path = 'data/processed/preprocessed_train_features.csv'
features.to_csv(preprocessed_features_path, index=False)

if 'generated' in data.columns:
    preprocessed_target_path = 'data/processed/preprocessed_train_target.csv'
    target.to_csv(preprocessed_target_path, index=False)

print("Preprocessing complete. Features and, if available, target variable saved.")
