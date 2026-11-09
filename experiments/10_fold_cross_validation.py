import pandas as pd
import mysql.connector
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import cross_val_score 
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# Connect to the database
connection = mysql.connector.connect(
    host="127.0.0.1", # MySQL adresa servera (localhost)
    user="root",  # MySQL korisničko ime
    password="Databejs567!",  # MySQL lozinka
    database="peptide-dataset"  # Naziv baze podataka
)
cursor = connection.cursor()

# Query the data
query = "SELECT * FROM `peptide-dataset`.peptides"
data = pd.read_sql(query, con=connection)

# Close the connection
connection.close()

# TODO:
# ROC/AUC, accuracy, F1 score, matthews correlation coefficient, precision, recall, confusion matrix
# promjeniti u modele u clasifikaciju (npr. RandomForestClassifier, LogisticRegression, KNeighborsClassifier, DecisionTreeClassifier)
# unutar cross validation napraviti feature selection

# Specify features (X) and target (y)
columns_to_drop = ["id", "peptide_seq", "target_col", "hydrophobic_cornette", "synthesis_flag"]
X = data.drop(columns=columns_to_drop)  # Replace 'target' with the actual target column name
y = data['target_col']  # Replace 'target' with the actual target column name

k = 10  # Number of folds for cross-validation
kf = KFold(n_splits=k, shuffle=True, random_state=42)


model = LinearRegression()

scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

average_r2 = np.mean(scores) 

print("/-----------------------------------------------------------------------------------------------------------------/")
print("10-Fold Cross-Validation Linear Regression R² Scores:")
print(f"R² Score for each fold: {[round(score, 4) for score in scores]}")
print(f"Average R² across {k} folds: {average_r2:.2f}")



# Initialize the Naive Bayes model
nb_model = GaussianNB()

# Perform 10-fold cross-validation
nb_scores = cross_val_score(nb_model, X, y, cv=kf, scoring='r2')

# Calculate the average R² score
average_nb_r2 = np.mean(nb_scores)

print("/-----------------------------------------------------------------------------------------------------------------/")
print("\n10-Fold Cross-Validation Naive Bayes R² Scores:")
print(f"R² Score for each fold: {[round(score, 4) for score in nb_scores]}")
print(f"Average R² across {k} folds: {average_nb_r2:.2f}")

# Initialize the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Perform 10-fold cross-validation
rf_scores = cross_val_score(rf_model, X, y, cv=kf, scoring='r2')

# Calculate the average R² score
average_rf_r2 = np.mean(rf_scores)

print("/-----------------------------------------------------------------------------------------------------------------/")
print("\n10-Fold Cross-Validation Random Forest R² Scores:")
print(f"R² Score for each fold: {[round(score, 4) for score in rf_scores]}")
print(f"Average R² across {k} folds: {average_rf_r2:.2f}")

# Initialize the Logistic Regression model
logistic_model = LogisticRegression(max_iter=1000)

# Perform 10-fold cross-validation
logistic_scores = cross_val_score(logistic_model, X, y, cv=kf, scoring='r2')

# Calculate the average R² score
average_logistic_r2 = np.mean(logistic_scores)

print("/-----------------------------------------------------------------------------------------------------------------/")
print("\n10-Fold Cross-Validation Logistic Regression R² Scores:")
print(f"R² Score for each fold: {[round(score, 4) for score in logistic_scores]}")
print(f"Average R² across {k} folds: {average_logistic_r2:.2f}")

# Initialize the K-Nearest Neighbors model
knn_model = KNeighborsRegressor(n_neighbors=5)

# Perform 10-fold cross-validation
knn_scores = cross_val_score(knn_model, X, y, cv=kf, scoring='r2')

# Calculate the average R² score
average_knn_r2 = np.mean(knn_scores)

print("/-----------------------------------------------------------------------------------------------------------------/")
print("\n10-Fold Cross-Validation K-Nearest Neighbors R² Scores:")
print(f"R² Score for each fold: {[round(score, 4) for score in knn_scores]}")
print(f"Average R² across {k} folds: {average_knn_r2:.2f}")

# Initialize the Support Vector Machine model
svm_model = SVR()

# Perform 10-fold cross-validation
svm_scores = cross_val_score(svm_model, X, y, cv=kf, scoring='r2')

# Calculate the average R² score
average_svm_r2 = np.mean(svm_scores)

print("/-----------------------------------------------------------------------------------------------------------------/")
print("\n10-Fold Cross-Validation Support Vector Machine R² Scores:")
print(f"R² Score for each fold: {[round(score, 4) for score in svm_scores]}")
print(f"Average R² across {k} folds: {average_svm_r2:.2f}")

# Initialize the Decision Tree model
dt_model = DecisionTreeRegressor(random_state=42)

# Perform 10-fold cross-validation
dt_scores = cross_val_score(dt_model, X, y, cv=kf, scoring='r2')

# Calculate the average R² score
average_dt_r2 = np.mean(dt_scores)

print("/-----------------------------------------------------------------------------------------------------------------/")
print("\n10-Fold Cross-Validation Decision Tree R² Scores:")
print(f"R² Score for each fold: {[round(score, 4) for score in dt_scores]}")
print(f"Average R² across {k} folds: {average_dt_r2:.2f}")


# Initialize the Random Forest model with different parameters
rf_model_2 = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)

# Perform 10-fold cross-validation
rf_scores_2 = cross_val_score(rf_model_2, X, y, cv=kf, scoring='r2')

# Calculate the average R² score
average_rf_r2_2 = np.mean(rf_scores_2)

print("/-----------------------------------------------------------------------------------------------------------------/")
print("\n10-Fold Cross-Validation Random Forest (n_estimators=200, max_depth=10) R² Scores:")
print(f"R² Score for each fold: {[round(score, 4) for score in rf_scores_2]}")
print(f"Average R² across {k} folds: {average_rf_r2_2:.2f}")




"""
/-----------------------------------------------------------------------------------------------------------------/
10-Fold Cross-Validation Linear Regression R² Scores:
R² Score for each fold: [np.float64(0.1257), np.float64(0.1762), np.float64(0.0684), np.float64(0.075), np.float64(0.1441),
np.float64(0.0796), np.float64(0.1102), np.float64(0.133), np.float64(0.0867), np.float64(0.1144)]
Average R² across 10 folds: 0.11

/-----------------------------------------------------------------------------------------------------------------/
10-Fold Cross-Validation Naive Bayes R² Scores:
R² Score for each fold: [np.float64(-0.5411), np.float64(-1.1988), np.float64(-1.5934), np.float64(-1.4239), np.float64(-0.3548),
np.float64(-0.8602), np.float64(-1.2987), np.float64(-1.1209), np.float64(-0.8034), np.float64(-1.2655)]
Average R² across 10 folds: -1.05

/-----------------------------------------------------------------------------------------------------------------/
10-Fold Cross-Validation Random Forest R² Scores:
R² Score for each fold: [np.float64(0.1345), np.float64(0.1143), np.float64(-0.0045), np.float64(-0.007), np.float64(0.1919),
np.float64(0.0718), np.float64(0.0795), np.float64(0.1178), np.float64(0.1247), np.float64(0.1746)]
Average R² across 10 folds: 0.10

/-----------------------------------------------------------------------------------------------------------------/
10-Fold Cross-Validation Logistic Regression R² Scores:
R² Score for each fold: [np.float64(-0.1558), np.float64(-0.1494), np.float64(-0.1346), np.float64(-0.071), np.float64(-0.0489),
np.float64(-0.071), np.float64(-0.0994), np.float64(-0.0123), np.float64(-0.1271), np.float64(0.0359)]
Average R² across 10 folds: -0.08

/-----------------------------------------------------------------------------------------------------------------/
10-Fold Cross-Validation K-Nearest Neighbors R² Scores:
R² Score for each fold: [np.float64(0.0561), np.float64(0.0945), np.float64(-0.1325), np.float64(-0.1229), np.float64(0.1067),
np.float64(-0.0575), np.float64(-0.0634), np.float64(-0.0778), np.float64(-0.0586), np.float64(-0.0123)]
Average R² across 10 folds: -0.03

/-----------------------------------------------------------------------------------------------------------------/
10-Fold Cross-Validation Support Vector Machine R² Scores:
R² Score for each fold: [np.float64(0.0239), np.float64(0.0543), np.float64(0.0076), np.float64(0.0117), np.float64(-0.0088), np.float64(0.0165),
np.float64(0.0038), np.float64(0.0406), np.float64(-0.0056), np.float64(0.0379)]
Average R² across 10 folds: 0.02

/-----------------------------------------------------------------------------------------------------------------/
10-Fold Cross-Validation Decision Tree R² Scores:
R² Score for each fold: [np.float64(-0.8301), np.float64(-1.0988), np.float64(-1.1612), np.float64(-0.6911), np.float64(-0.4422), np.float64(-0.5783),
np.float64(-0.9989), np.float64(-0.7835), np.float64(-0.7132), np.float64(-0.4461)]
Average R² across 10 folds: -0.77

/-----------------------------------------------------------------------------------------------------------------/
10-Fold Cross-Validation Random Forest (n_estimators=200, max_depth=10) R² Scores:
R² Score for each fold: [np.float64(0.1554), np.float64(0.2004), np.float64(0.0279), np.float64(0.021), np.float64(0.2016), np.float64(0.1094),
np.float64(0.1041), np.float64(0.1442), np.float64(0.1424), np.float64(0.1911)]
Average R² across 10 folds: 0.13
"""