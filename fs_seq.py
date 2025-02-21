import pandas as pd
import mysql.connector
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector

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

# Create a logistic regression model
logreg = LogisticRegression()

# Create a sequential feature selector
# Vec implementirana cross-validation
# promjeniti model npr. random forest
# i promjeniti scoring npr. accuracy, roc_auc, f1, precision, recall
selector = SequentialFeatureSelector(
	logreg, n_features_to_select=2, scoring='accuracy')

# Fit the selector to the data
selector.fit(X, y)

# Get the selected features
selected_features = selector.get_support()

print('The selected features are:', list(X.columns[selected_features]))

'''
The selected features are: ['isoelectric_point', 'polar_group']
'''
