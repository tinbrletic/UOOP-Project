import pandas as pd
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, matthews_corrcoef,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from skrebate import ReliefF
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

# Define classifiers with integrated feature selection.
# For SVM we integrate feature selection via SelectKBest inside the pipeline.
classifiers = {
    "Random Forest": {
        'model': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        'selector': SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), max_features=5)
    },
    "Logistic Regression": {
        'model': make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight='balanced',
                               solver='saga', penalty='l1', C=0.1,
                               random_state=42, n_jobs=-1)
        ),
        'selector': SelectFromModel(
            LogisticRegression(penalty='l1', solver='saga', max_iter=5000, random_state=42),
            max_features=5)
    }, # za svm knn i naiive bayes nismo direktno imali preko modela feature selection nego smo koristili vanjski selektor jer kod njih 
    "SVM": {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('selectkbest', SelectKBest(k=5)),
            ('svc', SVC(kernel='rbf', C=0.5, probability=True, random_state=42)) # c=0.5
        ]),
        'selector': None  # No external selector; use pipeline's internal selector.
    },
    "K-Neighbors": {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=15, algorithm='kd_tree'))
        ]),
        'selector': ReliefF(n_features_to_select=5, n_neighbors=20, n_jobs=-1)
    },
    "Decision Tree": {
        'model': DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42),
        'selector': SelectFromModel(DecisionTreeClassifier(random_state=42), max_features=5)
    },
    "Naive Bayes": {
        'model': make_pipeline(
            StandardScaler(),
            GaussianNB()
        ),
        'selector': SelectKBest(score_func=f_classif, k=5)
    }
}

# Configure cross-validation
#kf = KFold(n_splits=10, shuffle=True, random_state=42) # TODO: set repeated stratified kfold
kf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)

# Global feature importance dictionary for all classifiers
feature_importance_dict = {
    "Random Forest": pd.Series(0.0, index=X.columns),
    "Logistic Regression": pd.Series(0.0, index=X.columns),
    "SVM": pd.Series(0.0, index=X.columns),
    "K-Neighbors": pd.Series(0.0, index=X.columns),
    "Decision Tree": pd.Series(0.0, index=X.columns),
    "Naive Bayes": pd.Series(0.0, index=X.columns)
}

for clf_name, clf_info in classifiers.items():
    print(f"\n{'='*40}\nEvaluating {clf_name}\n{'='*40}")
    
    metrics_history = {metric: [] for metric in ['Accuracy', 'F1', 'AUC-ROC', 'Precision', 'Recall', 'MCC']}
    # Accumulate per-fold feature importance for this classifier.
    model_feature_importance = pd.Series(0.0, index=X.columns)
    confusion_matrices = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        if clf_info['selector'] is not None:
            # Use external selector branch.
            # For ReliefF (used with K-Neighbors), work with numpy arrays.
            if isinstance(clf_info['selector'], ReliefF):
                selector = clf_info['selector'].fit(X_train.values, y_train.values)
                X_train_selected = selector.transform(X_train.values)
                X_test_selected = selector.transform(X_test.values)
                selected_indices = selector.top_features_[:selector.n_features_to_select]
                selected_features = X.columns[selected_indices]
                model = clf_info['model'].fit(X_train_selected, y_train.values)
                relief_scores = selector.feature_importances_
                importances = pd.Series(relief_scores, index=X.columns).loc[selected_features]
            # For Naive Bayes, use SelectKBest and extract scores.
            elif clf_name == "Naive Bayes":
                selector = clf_info['selector'].fit(X_train, y_train)
                X_train_selected = selector.transform(X_train)
                X_test_selected = selector.transform(X_test)
                selected_features = X.columns[selector.get_support()]
                model = clf_info['model'].fit(X_train_selected, y_train)
                importances = pd.Series(selector.scores_[selector.get_support()],
                                        index=selected_features)
            else:
                selector = clf_info['selector'].fit(X_train, y_train)
                X_train_selected = selector.transform(X_train)
                X_test_selected = selector.transform(X_test)
                selected_features = X.columns[selector.get_support()]
                model = clf_info['model'].fit(X_train_selected, y_train)
                if clf_name == "Random Forest":
                    selected_mask = selector.get_support()
                    imp_values = selector.estimator_.feature_importances_[selected_mask]
                    importances = pd.Series(imp_values, index=selected_features)
                elif clf_name == "Logistic Regression":
                    coefs = model.named_steps['logisticregression'].coef_[0]
                    importances = pd.Series(np.abs(coefs), index=selected_features)
                elif clf_name == "Decision Tree":
                    imp_values = model.feature_importances_
                    importances = pd.Series(imp_values, index=selected_features)
                else:
                    importances = pd.Series(0.0, index=selected_features)
            full_importances = pd.Series(0.0, index=X.columns)
            full_importances[selected_features] = importances.values
            model_feature_importance += full_importances
        else:
            # For SVM, use the integrated selector in the pipeline.
            model = clf_info['model'].fit(X_train, y_train)
            internal_selector = model.named_steps['selectkbest']
            selected_mask = internal_selector.get_support()
            selected_features = X.columns[selected_mask]
            # Let the pipeline process full X_test.
            svm_model = model.named_steps['svc']

            importances = pd.Series(np.abs(internal_selector.scores_[selected_mask]), index=selected_features)
            full_importances = pd.Series(0.0, index=X.columns)
            full_importances[selected_features] = importances.values
            model_feature_importance += full_importances
            #model_feature_importance += None
            X_test_selected = X_test.values  # Pass numpy array to match pipeline's expected input.
        
        # For prediction, ensure the input matches the shape expected.
        y_pred = model.predict(X_test_selected)
        
        y_proba = model.predict_proba(X_test_selected)[:, 1]
        
        
        fold_metrics = calculate_metrics(y_test, y_pred, y_proba)
        for metric in metrics_history:
            metrics_history[metric].append(fold_metrics[metric])
        confusion_matrices.append(fold_metrics['Confusion_Matrix'])
    
    feature_importance_dict[clf_name] += model_feature_importance
    
    print("\nAverage Performance Metrics:")
    for metric in metrics_history:
        print(f"{metric}: {np.mean(metrics_history[metric]):.3f} ± {np.std(metrics_history[metric]):.3f}")
    
    print("\nTop 5 Features:")
    avg_importance = (model_feature_importance / 10).sort_values(ascending=False)
    for feat, imp in avg_importance.head(5).items():
        print(f"{feat}: {imp:.4f}")
    
    print("\nAggregated Confusion Matrix:")
    print(np.sum(confusion_matrices, axis=0))

print("\n" + "="*50)
print("Feature Importance Summary:")
for model_name, imp_series in feature_importance_dict.items():
    print(f"\n{model_name}:")
    print(imp_series.nlargest(5).to_string())



"""



========================================
Evaluating Random Forest
========================================

Average Performance Metrics:
Accuracy: 0.868 ± 0.013
F1: 0.928 ± 0.008
AUC-ROC: 0.721 ± 0.025
Precision: 0.882 ± 0.014
Recall: 0.979 ± 0.013
MCC: 0.205 ± 0.092

Top 5 Features:
hydrophobic_janin: 0.0306
hydrophobic_eisenberg: 0.0266     
hydrophobic_kyte-doolittle: 0.0214
charge: 0.0115
boman: 0.0091

Aggregated Confusion Matrix:      
[[  30  202]
 [  32 1507]]

========================================
Evaluating Logistic Regression
========================================

Average Performance Metrics:
Accuracy: 0.707 ± 0.043
F1: 0.809 ± 0.035
AUC-ROC: 0.705 ± 0.046
Precision: 0.925 ± 0.008
Recall: 0.721 ± 0.052
MCC: 0.241 ± 0.056

Top 5 Features:
non-polar_group: 0.7483
X8_PP: 0.1567
X8_IV: 0.1418
X8_GY: 0.1266
X4_H: 0.1241

Aggregated Confusion Matrix:
[[ 142   90]
 [ 429 1110]]

========================================
Evaluating SVM
========================================

Average Performance Metrics:
Accuracy: 0.869 ± 0.013
F1: 0.930 ± 0.007
AUC-ROC: 0.653 ± 0.071
Precision: 0.869 ± 0.013
Recall: 1.000 ± 0.000
MCC: 0.000 ± 0.000

Top 5 Features:
polar_group: 0.0012
boman: 0.0012
hydrophobic_engleman: 0.0009
aliphatic_index: 0.0007
non-polar_group: 0.0005

Aggregated Confusion Matrix:
[[   0  232]
 [   0 1539]]

========================================
Evaluating K-Neighbors
========================================

Average Performance Metrics:
Accuracy: 0.873 ± 0.014
F1: 0.931 ± 0.008
AUC-ROC: 0.692 ± 0.048
Precision: 0.882 ± 0.014
Recall: 0.986 ± 0.010
MCC: 0.214 ± 0.111

Top 5 Features:
non-polar_group: 0.1798
polar_group: 0.1291
X5_A: 0.1225
hydrophobic_janin: 0.1198
tiny_group: 0.0567

Aggregated Confusion Matrix:
[[  28  204]
 [  21 1518]]

========================================
Evaluating Decision Tree
========================================

Average Performance Metrics:
Accuracy: 0.863 ± 0.012
F1: 0.925 ± 0.007
AUC-ROC: 0.711 ± 0.034
Precision: 0.884 ± 0.014
Recall: 0.969 ± 0.007
MCC: 0.202 ± 0.060

Top 5 Features:
hydrophobic_janin: 0.3524
non-polar_group: 0.1373
peptide_len: 0.1068
isoelectric_point: 0.0886
hydrophobic_eisenberg: 0.0565

Aggregated Confusion Matrix:
[[  37  195]
 [  48 1491]]

========================================
Evaluating Naive Bayes
========================================

Average Performance Metrics:
Accuracy: 0.819 ± 0.020
F1: 0.894 ± 0.013
AUC-ROC: 0.734 ± 0.033
Precision: 0.911 ± 0.011
Recall: 0.878 ± 0.026
MCC: 0.281 ± 0.076

Top 5 Features:
non-polar_group: 141.3439
hydrophobic_janin: 133.5812
hydrophobic_kyte-doolittle: 127.8940
acidic_group: 114.8418
hydrophobic_eisenberg: 74.5839

Aggregated Confusion Matrix:
[[ 100  132]
 [ 188 1351]]

==================================================
Feature Importance Summary:

Random Forest:
hydrophobic_janin             0.305967
hydrophobic_eisenberg         0.266446
hydrophobic_kyte-doolittle    0.213549
charge                        0.114613
boman                         0.090978

Logistic Regression:
non-polar_group    7.482911
X8_PP              1.566854
X8_IV              1.417787
X8_GY              1.265853
X4_H               1.240733

SVM:
polar_group             0.012466
boman                   0.012466
hydrophobic_engleman    0.009440
aliphatic_index         0.006698
non-polar_group         0.005219

K-Neighbors:
non-polar_group      1.798132
polar_group          1.290985
X5_A                 1.224701
hydrophobic_janin    1.197976
tiny_group           0.567266

Decision Tree:
hydrophobic_janin        3.523718
non-polar_group          1.373169
peptide_len              1.067677
isoelectric_point        0.885713
hydrophobic_eisenberg    0.564833

Naive Bayes:
non-polar_group               1413.438901
hydrophobic_janin             1335.811507
hydrophobic_kyte-doolittle    1278.939943
acidic_group                  1148.417748
hydrophobic_eisenberg          745.839406
"""