import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load the preprocessed data
data_path = 'data/processed/preprocessed_train_data.csv'
data = pd.read_csv(data_path)

# Separate features (X) and target variable (y)
X = data.drop('generated', axis=1)  # Features
y = data['generated']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize and train a Random Forest classifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Model evaluation for Random Forest
rf_predictions = rf_model.predict(X_test)
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_predictions))

# Hyperparameter tuning for Random Forest
rf_param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20]}
rf_grid = GridSearchCV(RandomForestClassifier(), rf_param_grid, cv=5)
rf_grid.fit(X_train, y_train)

print("Best cross-validation score for Random Forest: ", rf_grid.best_score_)
print("Best parameters for Random Forest: ", rf_grid.best_params_)

# Initialize and train a Gradient Boosting classifier
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)

# Model evaluation for Gradient Boosting
gb_predictions = gb_model.predict(X_test)
print("\nGradient Boosting Classification Report:")
print(classification_report(y_test, gb_predictions))

# Hyperparameter tuning for Gradient Boosting
gb_param_grid = {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 1.0]}
gb_grid = GridSearchCV(GradientBoostingClassifier(), gb_param_grid, cv=5)
gb_grid.fit(X_train, y_train)

print("Best cross-validation score for Gradient Boosting: ", gb_grid.best_score_)
print("Best parameters for Gradient Boosting: ", gb_grid.best_params_)

# Initialize and train an SVM classifier
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Model evaluation for SVM
svm_predictions = svm_model.predict(X_test)
print("\nSVM Classification Report:")
print(classification_report(y_test, svm_predictions))

# Hyperparameter tuning for SVM
svm_param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}
svm_grid = GridSearchCV(SVC(), svm_param_grid, cv=5)
svm_grid.fit(X_train, y_train)

print("Best cross-validation score for SVM: ", svm_grid.best_score_)
print("Best parameters for SVM: ", svm_grid.best_params_)
