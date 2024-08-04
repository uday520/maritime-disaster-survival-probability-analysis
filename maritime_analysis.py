import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time

# Load the Titanic dataset from Seaborn
# titanic = sns.load_dataset("C:\\Users\\LENOVO\\Downloads\\Titanic-Dataset.csv")
import pandas as pd

# Specify the path to the CSV file
csv_file_path = "C:\\Users\\LENOVO\\Downloads\\Titanic-Dataset.csv"

# Read the CSV file into a DataFrame
titanic = pd.read_csv(csv_file_path)

# Fill missing values
titanic["Age"].fillna(titanic["Age"].median(), inplace=True)
titanic["Embarked"].fillna(titanic["Embarked"].mode()[0], inplace=True)

# Encode categorical variables
encoder = LabelEncoder()
titanic["Sex"] = encoder.fit_transform(titanic["Sex"])
titanic["Embarked"] = encoder.fit_transform(titanic["Embarked"])

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# Original Split (80-20)
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
    titanic[features], titanic['Survived'], test_size=0.2, random_state=42
)

# Model Names
model_names = ['Random Forest', 'Decision Tree', 'Logistic Regression']
models = [RandomForestClassifier(random_state=42), DecisionTreeClassifier(random_state=42),
          LogisticRegression(random_state=42)]

# Initialize lists to store results
precision_list_orig = []
recall_list_orig = []
f1_list_orig = []
accuracy_list_orig = []
time_taken_list_orig = []

# Original Split (80-20) Results
print("Original Split (80-20) Results:")
for model, name in zip(models, model_names):
    start_time = time.time()  # Start measuring time
    
    # Train the model
    model.fit(X_train_orig, y_train_orig)
    
    # Predictions
    y_pred = model.predict(X_test_orig)
    
    # Calculate metrics
    precision = precision_score(y_test_orig, y_pred)
    recall = recall_score(y_test_orig, y_pred)
    f1 = f1_score(y_test_orig, y_pred)
    accuracy = accuracy_score(y_test_orig, y_pred)
    
    end_time = time.time()  # Stop measuring time
    time_taken = (end_time - start_time) * 1e9  # Convert to nanoseconds
    
    # Append results to lists
    precision_list_orig.append(precision)
    recall_list_orig.append(recall)
    f1_list_orig.append(f1)
    accuracy_list_orig.append(accuracy)
    time_taken_list_orig.append(time_taken)
    
    # Print Results
    print(f"{name} Results (80-20 Split):")
    print(f"Accuracy: {accuracy:.2f}")
    
    # Print Confusion Matrix to Console
    print(f"Confusion Matrix:\n{confusion_matrix(y_test_orig, y_pred)}\n")

    # Plotting the Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test_orig, y_pred), annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{name} Confusion Matrix (80-20)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# 70-30 Split Results
precision_list_70_30 = []
recall_list_70_30 = []
f1_list_70_30 = []
accuracy_list_70_30 = []
time_taken_list_70_30 = []

print("\n70-30 Split Results:")
for model, name in zip(models, model_names):
    start_time = time.time()  # Start measuring time
    
    # Split data into 70-30 ratio
    X_train, X_test, y_train, y_test = train_test_split(
        titanic[features], titanic['Survived'], test_size=0.3, random_state=42
    )

    # Train the model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    end_time = time.time()  # Stop measuring time
    time_taken = (end_time - start_time) * 1e9  # Convert to nanoseconds
    
    # Append results to lists
    precision_list_70_30.append(precision)
    recall_list_70_30.append(recall)
    f1_list_70_30.append(f1)
    accuracy_list_70_30.append(accuracy)
    time_taken_list_70_30.append(time_taken)
    
    # Print Results
    print(f"{name} Results (70-30 Split):")
    print(f"Accuracy: {accuracy:.2f}")
    
    # Print Confusion Matrix to Console
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")

    # Plotting the Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{name} Confusion Matrix (70-30)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# 75-25 Split Results
precision_list_75_25 = []
recall_list_75_25 = []
f1_list_75_25 = []
accuracy_list_75_25 = []
time_taken_list_75_25 = []

print("\n75-25 Split Results:")
for model, name in zip(models, model_names):
    start_time = time.time()  # Start measuring time
    
    # Split data into 75-25 ratio
    X_train, X_test, y_train, y_test = train_test_split(
        titanic[features], titanic['Survived'], test_size=0.25, random_state=42
    )

    # Train the model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    end_time = time.time()  # Stop measuring time
    time_taken = (end_time - start_time) * 1e9  # Convert to nanoseconds
    
    # Append results to lists
    precision_list_75_25.append(precision)
    recall_list_75_25.append(recall)
    f1_list_75_25.append(f1)
    accuracy_list_75_25.append(accuracy)
    time_taken_list_75_25.append(time_taken)
    
    # Print Results
    print(f"{name} Results (75-25 Split):")
    print(f"Accuracy: {accuracy:.2f}")
    
    # Print Confusion Matrix to Console
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")

    # Plotting the Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{name} Confusion Matrix (75-25)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Prepare comparison tables
comparison_table_orig = pd.DataFrame({
    'Model Name': model_names,
    'Precision (80-20)': precision_list_orig,
    'Recall (80-20)': recall_list_orig,
    'F1 Score (80-20)': f1_list_orig,
    'Accuracy (80-20)': accuracy_list_orig,
    'Time Taken (ns) (80-20)': time_taken_list_orig
})

comparison_table_70_30 = pd.DataFrame({
    'Model Name': model_names,
    'Precision (70-30)': precision_list_70_30,
    'Recall (70-30)': recall_list_70_30,
    'F1 Score (70-30)': f1_list_70_30,
    'Accuracy (70-30)': accuracy_list_70_30,
    'Time Taken (ns) (70-30)': time_taken_list_70_30
})

comparison_table_75_25 = pd.DataFrame({
    'Model Name': model_names,
    'Precision (75-25)': precision_list_75_25,
    'Recall (75-25)': recall_list_75_25,
    'F1 Score (75-25)': f1_list_75_25,
    'Accuracy (75-25)': accuracy_list_75_25,
    'Time Taken (ns) (75-25)': time_taken_list_75_25
})

# Print the comparison tables
print("\nComparison Table (80-20):")
print(comparison_table_orig)

print("\nComparison Table (70-30):")
print(comparison_table_70_30)

print("\nComparison Table (75-25):")
print(comparison_table_75_25)
