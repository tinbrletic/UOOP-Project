import pandas as pd
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, 
                            precision_score, recall_score, matthews_corrcoef,
                            confusion_matrix)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectFromModel
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def calculate_metrics(y_true, y_pred, y_proba):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'AUC-ROC': roc_auc_score(y_true, y_proba),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'Confusion_Matrix': confusion_matrix(y_true, y_pred)
    }

# Load and prepare data
data = pd.read_csv('peptide_baza_formatted.csv', sep=';', quotechar='"')
columns_to_drop = ["id", "peptide_seq", "targetcol", "hydrophobic_cornette", "synthesis_flag"]
X = data.drop(columns=columns_to_drop)
y = data['targetcol']

# Configure classifiers with integrated feature selection
classifiers = {
    "Random Forest": {
        'model': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        'selector': SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=5)
    },
    "Logistic Regression": {
        'model': make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight='balanced', 
                             solver='saga', penalty='l1', C=0.1, random_state=42, n_jobs=-1)
        ),
        'selector': SelectFromModel(LogisticRegression(), max_features=5)
    },
    "SVM": {
        'model': make_pipeline(
            StandardScaler(),
            SVC(kernel='linear', probability=True, class_weight='balanced',
                C=0.5, max_iter=1000, shrinking=True, random_state=42)
        ),
        'selector': SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=5)
    }
}

kf = KFold(n_splits=10, shuffle=True, random_state=42)

for clf_name, clf_info in classifiers.items():
    print(f"\n{'='*40}\nEvaluating {clf_name}\n{'='*40}")
    
    metrics_history = {metric: [] for metric in ['Accuracy', 'F1', 'AUC-ROC', 'Precision', 'Recall', 'MCC']}
    feature_importance = pd.Series(0.0, index=X.columns)
    confusion_matrices = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Feature selection
        selector = clf_info['selector'].fit(X_train, y_train)
        X_train_selected = selector.transform(X_train)
        X_test_selected = selector.transform(X_test)
        
        # Get selected feature indices
        selected_features = X.columns[selector.get_support()]
        
        # Model training
        model = clf_info['model'].fit(X_train_selected, y_train)
        
        # Feature importance extraction
        if clf_name == "Random Forest":
            importances = pd.Series(model.feature_importances_, index=selected_features)
        elif clf_name == "Logistic Regression":
            importances = pd.Series(np.abs(model.named_steps['logisticregression'].coef_[0]), index=selected_features)
        else:  # SVM
            importances = pd.Series(np.abs(model.named_steps['svc'].coef_[0]), index=selected_features)
        
        # Update feature importance
        full_importances = pd.Series(0.0, index=X.columns)
        full_importances[selected_features] = importances
        feature_importance += full_importances
        
        # Predictions
        y_pred = model.predict(X_test_selected)
        y_proba = model.predict_proba(X_test_selected)[:,1] if hasattr(model, "predict_proba") else model.decision_function(X_test_selected)
        
        # Store metrics
        fold_metrics = calculate_metrics(y_test, y_pred, y_proba)
        for metric in metrics_history:
            metrics_history[metric].append(fold_metrics[metric])
        confusion_matrices.append(fold_metrics['Confusion_Matrix'])
    
    # Print performance metrics
    print("\nAverage Performance Metrics:")
    for metric in metrics_history:
        print(f"{metric}: {np.mean(metrics_history[metric]):.3f} ± {np.std(metrics_history[metric]):.3f}")

    # Print top 5 features
    print("\nTop 5 Most Important Features:")
    avg_importance = (feature_importance / 10).sort_values(ascending=False)
    for feature, importance in avg_importance.head(5).items():
        print(f"{feature}: {importance:.4f}")

    # Print confusion matrix
    print("\nAggregated Confusion Matrix:")
    print(np.sum(confusion_matrices, axis=0))
