import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, matthews_corrcoef
)

# Load and prepare data
data = pd.read_csv('peptide_baza_formatted.csv', sep=';', quotechar='"')
columns_to_drop = ["id", "peptide_seq", "targetcol", "hydrophobic_cornette", "synthesis_flag"]
X = data.drop(columns=columns_to_drop)
y = data['targetcol']

# Define models with your specified parameters
models = {
    'Logistic Regression': LogisticRegression(max_iter=5000, class_weight='balanced', solver='saga', penalty='l1', C=0.1, random_state=42, n_jobs=-1),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    'SVM': SVC(kernel='linear', C=0.5, probability=True, random_state=42),
    'K-Neighbors': KNeighborsClassifier(n_neighbors=15, algorithm='kd_tree'),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42),
    'Naive Bayes': GaussianNB()
}

# Create a mapping for scoring metrics:
# Key: metric label for printing; Value: valid scoring string for SequentialFeatureSelector
scoring_metrics = {
    'accuracy': 'accuracy',
    'f1': 'f1',
    'roc_auc': 'roc_auc',
    'precision': 'precision',
    'recall': 'recall',
    'mcc': 'matthews_corrcoef'
}

# Evaluate each model with each scoring metric
for model_name, model in models.items():
    for metric_name, scoring in scoring_metrics.items():
        # Create the Sequential Feature Selector using the valid scoring identifier
        selector = SequentialFeatureSelector(
            model,
            n_features_to_select=5,
            scoring=scoring,
            cv=10,
            n_jobs=-1
        )
        selector.fit(X, y)
        selected_features = list(X.columns[selector.get_support()])
        
        # Split data using only the selected features
        X_selected = X[selected_features]
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
        
        # Train the model on the train set and predict on the test set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Compute the evaluation score
        if metric_name == 'accuracy':
            score = accuracy_score(y_test, y_pred)
        elif metric_name == 'f1':
            score = f1_score(y_test, y_pred, average='weighted')
        elif metric_name == 'roc_auc':
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
                score = roc_auc_score(y_test, y_proba)
            else:
                score = "Not applicable"
        elif metric_name == 'precision':
            score = precision_score(y_test, y_pred, average='weighted')
        elif metric_name == 'recall':
            score = recall_score(y_test, y_pred, average='weighted')
        elif metric_name == 'mcc':
            score = matthews_corrcoef(y_test, y_pred)
        else:
            score = "Unsupported metric"
            
        # Print the outputs: model name, scoring metric name, score result and selected features
        print(f"\nModel: {model_name} | Scoring: {metric_name}")
        print(f"{metric_name.upper()} Score: {score}")
        print(f"Selected features: {selected_features}")
