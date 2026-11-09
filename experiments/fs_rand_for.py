import pandas as pd
import numpy as np
import mysql.connector
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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

# Specify features (X) and target (y)
columns_to_drop = ["id", "peptide_seq", "target_col", "hydrophobic_cornette", "synthesis_flag"]
X = data.drop(columns=columns_to_drop)  # Replace with your feature columns
y = data["target_col"]  # Replace with your target column


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate the model
accuracy_before = rf.score(X_test, y_test)
print(f'Accuracy before feature selection: {accuracy_before:.2f}')

# Extract feature importances
importances = rf.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Rank features by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)

# Select top N features (example selecting top 10 features)
top_features = feature_importance_df['Feature'][:10].values
X_train_selected = X_train[top_features]
X_test_selected = X_test[top_features]

# Train the Random Forest model with selected features
rf_selected = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selected.fit(X_train_selected, y_train)

# Evaluate the model
accuracy_after = rf_selected.score(X_test_selected, y_test)
print(f'Accuracy after feature selection: {accuracy_after:.2f}')


'''
Accuracy before feature selection: 0.88
                       Feature  Importance
1            hydrophobic_janin    0.058957
5            isoelectric_point    0.055282
19           instability_index    0.054985
18               cruciani_prp3    0.052960
6                       charge    0.052187
23       hydrophobic_eisenberg    0.051407
17               cruciani_prp2    0.051250
3           hydrophobic_moment    0.050785
24         hydrophobic_roseman    0.050394
2         hydrophobic_engleman    0.050062
20                       boman    0.049794
21  hydrophobic_kyte-doolittle    0.047832
16               cruciani_prp1    0.046914
22      hydrophobic_hopp-woods    0.040077
4              aliphatic_index    0.037093
12                 polar_group    0.037013
11             non-polar_group    0.031865
8                  small_group    0.028116
10              aromatic_group    0.024782
0                  peptide_len    0.023817
15                acidic_group    0.023561
9              aliphatic_group    0.023556
7                   tiny_group    0.022973
14                 basic_group    0.020583
13               charged_group    0.013756
Accuracy after feature selection: 0.87
'''