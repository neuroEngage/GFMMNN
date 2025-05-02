# Importing necessary libraries
import pandas as pd
import numpy as np

# Visualization (Optional if you want plots)
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer  # Import the imputer

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# To save models and scaler
import joblib

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# 1. Load the dataset
df = pd.read_csv('heart_disease_uci.csv')

# 2. Quick check on data
print("Dataset shape:", df.shape)
print(df.head())

# 3. Drop 'id' column if it exists (not useful for prediction)
if 'id' in df.columns:
    df.drop(columns=['id'], inplace=True)

# 4. Encode / Convert categorical columns

# Encode 'sex': Map Male -> 1, Female -> 0
if 'sex' in df.columns:
    df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})

# Drop 'dataset' if it is constant (e.g., always "Cleveland")
if 'dataset' in df.columns:
    if df['dataset'].nunique() == 1:
        df.drop(columns=['dataset'], inplace=True)
    else:
        le_dataset = LabelEncoder()
        df['dataset'] = le_dataset.fit_transform(df['dataset'])

# Encode 'cp' (chest pain type) using LabelEncoder
if 'cp' in df.columns:
    le_cp = LabelEncoder()
    df['cp'] = le_cp.fit_transform(df['cp'])

# Convert 'fbs' (fasting blood sugar) from string to numeric (TRUE -> 1, FALSE -> 0)
if 'fbs' in df.columns:
    df['fbs'] = df['fbs'].map({'TRUE': 1, 'FALSE': 0})

# Encode 'restecg' (resting electrocardiographic results) using LabelEncoder
if 'restecg' in df.columns:
    le_restecg = LabelEncoder()
    df['restecg'] = le_restecg.fit_transform(df['restecg'])

# Convert 'exang' (exercise-induced angina) from string to numeric (TRUE -> 1, FALSE -> 0)
if 'exang' in df.columns:
    df['exang'] = df['exang'].map({'TRUE': 1, 'FALSE': 0})

# Encode 'slope' column using LabelEncoder (if not already numeric)
if 'slope' in df.columns:
    le_slope = LabelEncoder()
    df['slope'] = le_slope.fit_transform(df['slope'])

# Encode 'thal' column using LabelEncoder
if 'thal' in df.columns:
    le_thal = LabelEncoder()
    df['thal'] = le_thal.fit_transform(df['thal'])

# 5. Split Features and Target
# In your dataset the target column is 'num'
X = df.drop(columns=['num'])
y = df['num']

# Impute missing values in X using SimpleImputer with median strategy
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler to disk
joblib.dump(scaler, "scaler.pkl")
print("Scaler saved as 'scaler.pkl'")

# 8. Create a list of models
models = [
    ('Logistic Regression', LogisticRegression()),
    ('K-Nearest Neighbors', KNeighborsClassifier()),
    ('Support Vector Machine', SVC(probability=True)),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier()),
    ('Extra Trees', ExtraTreesClassifier()),
    ('AdaBoost', AdaBoostClassifier()),
    ('Gradient Boosting', GradientBoostingClassifier()),
    ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
]

# Dictionaries to store evaluation results and trained models
results = {}
trained_models = {}

# 9. Train and Evaluate Models (but do not save individually)
for name, model in models:
    print(f"\n{'=' * 30}\nTraining Model: {name}")

    # Train the model
    model.fit(X_train_scaled, y_train)

    # Store the trained model
    trained_models[name] = model

    # Predict on test set
    y_pred = model.predict(X_test_scaled)

    # Evaluate the model
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Store results
    results[name] = {'Accuracy': acc, 'Confusion Matrix': cm, 'Classification Report': report}

    # Print evaluation summary
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)

# 10. Determine the Best Model Based on Accuracy
best_model_name = max(results, key=lambda name: results[name]['Accuracy'])
best_accuracy = results[best_model_name]['Accuracy']
best_model = trained_models[best_model_name]

print("\n\n=== Best Model Summary ===")
print(f"Best Model: {best_model_name}")
print(f"Accuracy: {best_accuracy:.4f}")
print("Confusion Matrix:\n", results[best_model_name]['Confusion Matrix'])
print("Classification Report:\n", results[best_model_name]['Classification Report'])

# 11. Save only the best model
joblib.dump(best_model, "best_model.pkl")
print(f"\nBest model saved as 'best_model.pkl'")

# 12. (Optional) Create a summary table of accuracies for all models
accuracy_df = pd.DataFrame({name: [results[name]['Accuracy']] for name in results}).T
accuracy_df.columns = ['Accuracy']
accuracy_df.sort_values(by='Accuracy', ascending=False, inplace=True)
print("\n=== Accuracy Summary   for All Models ===")
print(accuracy_df)
