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
        n_jobs=-1
    ),
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=5000,
            class_weight='balanced',
            solver='saga',
            penalty='l1',
            C=0.1,
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
            C=0.5,
            max_iter=1000,
            shrinking=True
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
        
        # Feature selection and importance handling for SVM
        if clf_name == "SVM":
            # Preselect features using Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            top_features = X.columns[np.argsort(rf.feature_importances_)[-50:]]
            
            # Train SVM on selected features
            fitted_model = clf.fit(X_train[top_features], y_train)
            
            # Extract coefficients and map to original features
            coefs = np.abs(fitted_model.named_steps['svc'].coef_[0])
            importances = pd.Series(0.0, index=X.columns)
            importances[top_features] = coefs.astype(float)  # Explicit float conversion
            
            # Make predictions on modified test set
            y_pred = clf.predict(X_test[top_features])
            y_proba = clf.decision_function(X_test[top_features])
            
        else:
            # Standard training for other models
            fitted_model = clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else [0]*len(y_test)
            
            # Extract feature importances
            if clf_name == "Random Forest":
                importances = fitted_model.feature_importances_
            elif clf_name == "Logistic Regression":
                importances = np.abs(fitted_model.named_steps['logisticregression'].coef_[0])
        
        # Store feature importances (handle SVM separately)
        if clf_name != "SVM":
            feature_importance[clf_name] += pd.Series(importances, index=X.columns)
        else:
            feature_importance[clf_name] += importances
        
        # Handle probability scaling for SVM
        if clf_name == "SVM":
            y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
        
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