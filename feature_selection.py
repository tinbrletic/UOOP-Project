import pandas as pd
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import warnings

# Suppress convergence warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Load and prepare data
data = pd.read_csv('peptide_baza_formatted.csv', sep=';', quotechar='"')
columns_to_drop = ["id", "peptide_seq", "targetcol", "hydrophobic_cornette", "synthesis_flag"]
X = data.drop(columns=columns_to_drop)
y = data['targetcol']

# Configure classifiers with optimized parameters
classifiers = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1  # Enable parallel processing
    ),
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=5000,
            class_weight='balanced',
            solver='saga',  # Better for large datasets
            penalty='l1',  # L1 regularization for feature selection
            C=0.1,  # Stronger regularization
            random_state=42,
            n_jobs=-1
        )
    ),
    "SVM": make_pipeline(
        StandardScaler(),
        SVC(
            kernel='linear',
            probability=True,
            class_weight='balanced',
            random_state=42,
            C=0.5,  # Regularization parameter
            max_iter=1000,  # Increased iteration limit
            shrinking=True  # Enable shrinking heuristic
        )
    )
}

# Configure cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Store feature importance rankings
feature_importance = {
    "Random Forest": pd.Series(0, index=X.columns),
    "Logistic Regression": pd.Series(0, index=X.columns),
    "SVM": pd.Series(0, index=X.columns)
}

for clf_name, clf in classifiers.items():
    print(f"\n{'='*40}\nEvaluating {clf_name}\n{'='*40}")
    
    fold_metrics = {'Accuracy': [], 'F1-Score': [], 'AUC-ROC': []}
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        

        # TODO: ADD THE MODIFIED SVM SECTION
        # Feature selection for SVM
        if clf_name == "SVM":
            # Pre-select top 50 features using Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            top_features = X.columns[np.argsort(rf.feature_importances_)[-50:]]
            X_train = X_train[top_features]
            X_test = X_test[top_features]
        
        # Model training
        fitted_model = clf.fit(X_train, y_train)
        
        # Feature importance extraction
        if clf_name == "Random Forest":
            importances = fitted_model.feature_importances_
        elif clf_name == "Logistic Regression":
            importances = np.abs(fitted_model.named_steps['logisticregression'].coef_[0])
        else:
            importances = np.abs(fitted_model.named_steps['svc'].coef_[0])
        
        # Store feature importances
        feature_importance[clf_name] += pd.Series(importances, index=X.columns)
        
        # Predictions
        y_pred = clf.predict(X_test)
        
        # Handle probability estimation for SVM
        if clf_name == "SVM":
            y_proba = clf.decision_function(X_test)
            y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
        else:
            y_proba = clf.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        fold_metrics['Accuracy'].append(accuracy_score(y_test, y_pred))
        fold_metrics['F1-Score'].append(f1_score(y_test, y_pred))
        fold_metrics['AUC-ROC'].append(roc_auc_score(y_test, y_proba))

    # Print performance metrics
    print(f"Average Accuracy: {np.mean(fold_metrics['Accuracy']):.3f}")
    print(f"Average F1-score: {np.mean(fold_metrics['F1-Score']):.3f}")
    print(f"Average AUC-ROC: {np.mean(fold_metrics['AUC-ROC']):.3f}")
    
    # Print top 5 features
    print("\nTop 5 Features:")
    avg_importance = feature_importance[clf_name] / 10
    for feature, importance in avg_importance.nlargest(5).items():
        print(f"{feature}: {importance:.4f}")

print("\n" + "="*50)
print("Feature Importance Summary:")
for model in classifiers:
    print(f"\n{model}:")
    print(feature_importance[model].nlargest(5).to_string())
