import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the enhanced dataset
data_path = 'data/processed/enhanced_train_essays.csv'
data = pd.read_csv(data_path)

# Identify non-numeric columns for one-hot encoding 

non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns.tolist()

# Drop columns that are non-numeric and not useful for the model (if any)
# Example: data = data.drop(['unnecessary_column'], axis=1)

# Set up preprocessing for categorical data (one-hot encoding)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), non_numeric_columns)
    ], remainder='passthrough'
)

# Assume 'target_column' is your  target variable 
# Seperate features and target variable
X = data.drop('generated', axis=1) # Features
y = data['generated'] # Target variable

# Preprocess the data using for non-numeric columns using ColumnTransformer
X = preprocessor.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialise and train a logistic regression model
# Chosen for its simplicity and effectivness as a baseline model
model = LogisticRegression()
model.fit(X_train, y_train)

# Model evaluation
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

# Hyperparameter tuning 
param_grid = {'C': [0.01,0.1,1,10,100]}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)

print("Best cross-validation score: ", grid.best_score_)
print("Best parameters: ", grid.best_params_)
      
