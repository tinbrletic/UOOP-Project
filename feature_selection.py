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
import datetime
import os

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def log_configuration_info():
    """Log detailed configuration information for reproducibility"""
    print("=" * 80)
    print("FEATURE SELECTION AND CLASSIFICATION EXPERIMENT")
    print("=" * 80)
    print(f"Experiment started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {os.getcwd()}")
    print()
    
    # Dataset information
    dataset_file = 'peptide_baza_formatted.csv'
    if os.path.exists(dataset_file):
        file_size = os.path.getsize(dataset_file) / (1024 * 1024)  # MB
        modification_time = datetime.datetime.fromtimestamp(os.path.getmtime(dataset_file))
        print("DATASET INFORMATION:")
        print("-" * 50)
        print(f"Dataset file: {dataset_file}")
        print(f"File size: {file_size:.2f} MB")
        print(f"Last modified: {modification_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load and analyze dataset
        data = pd.read_csv(dataset_file, sep=';', quotechar='"')
        print(f"Dataset shape: {data.shape}")
        print(f"Target distribution:")
        target_counts = data['targetcol'].value_counts().sort_index()
        for target, count in target_counts.items():
            percentage = (count / len(data)) * 100
            print(f"  Class {target}: {count:,} samples ({percentage:.1f}%)")
        
        # Check for missing values
        missing_values = data.isnull().sum().sum()
        print(f"Missing values: {missing_values}")
        
        return data
    else:
        print(f"ERROR: Dataset file '{dataset_file}' not found!")
        return None

def log_preprocessing_info(data, columns_to_drop):
    """Log preprocessing information"""
    print()
    print("PREPROCESSING CONFIGURATION:")
    print("-" * 50)
    print(f"Columns to drop: {columns_to_drop}")
    
    X = data.drop(columns=columns_to_drop)
    y = data['targetcol']
    
    print(f"Features after preprocessing: {X.shape[1]}")
    print(f"Feature names (first 10): {list(X.columns[:10])}")
    if len(X.columns) > 10:
        print(f"... and {len(X.columns) - 10} more features")
    
    print(f"Target variable: 'targetcol'")
    print(f"Target shape: {y.shape}")
    
    return X, y

def log_cv_configuration(cv_strategy):
    """Log cross-validation configuration"""
    print()
    print("CROSS-VALIDATION CONFIGURATION:")
    print("-" * 50)
    if isinstance(cv_strategy, RepeatedStratifiedKFold):
        print(f"Strategy: Repeated Stratified K-Fold")
        print(f"Number of folds: {cv_strategy.cvargs['n_splits']}")
        print(f"Number of repeats: {cv_strategy.n_repeats}")
        print(f"Total CV iterations: {cv_strategy.cvargs['n_splits'] * cv_strategy.n_repeats}")
        print(f"Random state: {cv_strategy.random_state}")
    elif isinstance(cv_strategy, KFold):
        print(f"Strategy: K-Fold")
        print(f"Number of folds: {cv_strategy.n_splits}")
        print(f"Shuffle: {cv_strategy.shuffle}")
        print(f"Random state: {cv_strategy.random_state}")
    else:
        print(f"Strategy: {type(cv_strategy).__name__}")

def log_classifier_configuration(classifiers):
    """Log classifier configurations"""
    print()
    print("CLASSIFIER CONFIGURATIONS:")
    print("-" * 50)
    
    for clf_name, clf_info in classifiers.items():
        print(f"\n{clf_name}:")
        print(f"  Model: {type(clf_info['model']).__name__}")
        
        # Extract model parameters
        if hasattr(clf_info['model'], 'get_params'):
            params = clf_info['model'].get_params()
            key_params = {}
            
            # Filter out common parameters and focus on key ones
            for key, value in params.items():
                if key in ['n_estimators', 'max_depth', 'C', 'kernel', 'n_neighbors', 
                          'algorithm', 'penalty', 'solver', 'max_iter', 'class_weight',
                          'min_samples_split', 'random_state', 'n_jobs']:
                    key_params[key] = value
            
            if key_params:
                print(f"  Key parameters: {key_params}")
        
        # Feature selection info
        if clf_info['selector'] is not None:
            selector_type = type(clf_info['selector']).__name__
            print(f"  Feature selector: {selector_type}")
            if hasattr(clf_info['selector'], 'get_params'):
                selector_params = clf_info['selector'].get_params()
                key_selector_params = {}
                for key, value in selector_params.items():
                    if key in ['k', 'max_features', 'n_features_to_select', 'score_func',
                              'n_neighbors', 'n_jobs']:
                        key_selector_params[key] = value
                if key_selector_params:
                    print(f"  Selector parameters: {key_selector_params}")
        else:
            print(f"  Feature selector: Integrated in pipeline")

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

# Log configuration and load data
data = log_configuration_info()
if data is None:
    exit(1)

# Load and prepare data with logging
columns_to_drop = ["id", "peptide_seq", "targetcol", "hydrophobic_cornette", "synthesis_flag"]
X, y = log_preprocessing_info(data, columns_to_drop)

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
kf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)

# Log cross-validation configuration
log_cv_configuration(kf)

# Log classifier configurations
log_classifier_configuration(classifiers)

# Global feature importance dictionary for all classifiers
feature_importance_dict = {
    "Random Forest": pd.Series(0.0, index=X.columns),
    "Logistic Regression": pd.Series(0.0, index=X.columns),
    "SVM": pd.Series(0.0, index=X.columns),
    "K-Neighbors": pd.Series(0.0, index=X.columns),
    "Decision Tree": pd.Series(0.0, index=X.columns),
    "Naive Bayes": pd.Series(0.0, index=X.columns)
}

# Initialize results dictionary to store all classifier performance
all_classifier_results = {}

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
    
    # Calculate average metrics and store results
    avg_metrics = {}
    std_metrics = {}
    for metric in metrics_history:
        avg_metrics[metric] = np.mean(metrics_history[metric])
        std_metrics[metric] = np.std(metrics_history[metric])
    
    # Store results for this classifier
    all_classifier_results[clf_name] = {
        'avg_metrics': avg_metrics,
        'std_metrics': std_metrics,
        'top_features': (model_feature_importance / kf.cvargs['n_splits']).sort_values(ascending=False).head(10),
        'confusion_matrix': np.sum(confusion_matrices, axis=0),
        'total_cv_folds': len(metrics_history['Accuracy'])
    }
    
    print("\nAverage Performance Metrics:")
    for metric in metrics_history:
        print(f"{metric}: {avg_metrics[metric]:.3f} ± {std_metrics[metric]:.3f}")
    
    print("\nTop 5 Features:")
    avg_importance = (model_feature_importance / kf.cvargs['n_splits']).sort_values(ascending=False)
    for feat, imp in avg_importance.head(5).items():
        print(f"{feat}: {imp:.4f}")
    
    print("\nAggregated Confusion Matrix:")
    print(all_classifier_results[clf_name]['confusion_matrix'])

print("\n" + "="*80)
print("EXPERIMENT RESULTS SUMMARY")
print("="*80)
print(f"Experiment completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Dataset: peptide_baza_balanced.csv")
print(f"Total samples: {len(y)}")
print(f"Features used: {X.shape[1]}")
print(f"Cross-validation: {kf.cvargs['n_splits']}-fold, {kf.n_repeats} repeats")
print()

print("PERFORMANCE SUMMARY BY CLASSIFIER:")
print("-" * 80)

# Display results for each classifier
for clf_name, results in all_classifier_results.items():
    print(f"\n{clf_name}:")
    print(f"  Total CV folds: {results['total_cv_folds']}")
    for metric, avg_val in results['avg_metrics'].items():
        std_val = results['std_metrics'][metric]
        print(f"  {metric}: {avg_val:.3f} ± {std_val:.3f}")
    print(f"  Top 3 features: {', '.join(results['top_features'].head(3).index.tolist())}")

print("\nFEATURE IMPORTANCE SUMMARY:")
print("-" * 50)
for model_name, imp_series in feature_importance_dict.items():
    print(f"\n{model_name} - Top 5 Features:")
    for feat, imp in imp_series.nlargest(5).items():
        print(f"  {feat}: {imp:.4f}")

# Save comprehensive results with timestamp
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
results_filename = f"feature_selection_results_{timestamp}.txt"

with open(results_filename, 'w') as f:
    f.write("PEPTIDE SYNTHESIS PREDICTION - FEATURE SELECTION RESULTS\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Experiment Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Dataset: peptide_baza_balanced.csv\n")
    f.write(f"Total samples: {len(y)}\n")
    f.write(f"Total features: {X.shape[1]}\n")
    f.write(f"Cross-validation: {kf.cvargs['n_splits']}-fold, {kf.n_repeats} repeats\n")
    f.write(f"Random state: {kf.random_state}\n\n")
    
    # Write detailed performance metrics for each classifier
    f.write("DETAILED PERFORMANCE METRICS:\n")
    f.write("=" * 80 + "\n\n")
    
    for clf_name, results in all_classifier_results.items():
        f.write(f"{clf_name}:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total CV iterations: {results['total_cv_folds']}\n\n")
        
        # Performance metrics
        f.write("Performance Metrics:\n")
        for metric, avg_val in results['avg_metrics'].items():
            std_val = results['std_metrics'][metric]
            f.write(f"  {metric}: {avg_val:.6f} ± {std_val:.6f}\n")
        
        # Confusion matrix
        f.write(f"\nConfusion Matrix (Aggregated):\n")
        cm = results['confusion_matrix']
        f.write(f"  True Negative:  {cm[0,0]:6d}    False Positive: {cm[0,1]:6d}\n")
        f.write(f"  False Negative: {cm[1,0]:6d}    True Positive:  {cm[1,1]:6d}\n")
        
        # Top features
        f.write(f"\nTop 10 Features (Average Importance):\n")
        for i, (feat, imp) in enumerate(results['top_features'].head(10).items(), 1):
            f.write(f"  {i:2d}. {feat:<30} {imp:.6f}\n")
        
        f.write("\n" + "="*80 + "\n\n")
    
    # Summary comparison table
    f.write("CLASSIFIER COMPARISON SUMMARY:\n")
    f.write("=" * 80 + "\n")
    f.write(f"{'Classifier':<20} {'Accuracy':<12} {'F1-Score':<12} {'Precision':<12} {'Recall':<12} {'AUC-ROC':<12}\n")
    f.write("-" * 80 + "\n")
    
    for clf_name, results in all_classifier_results.items():
        metrics = results['avg_metrics']
        f.write(f"{clf_name:<20} "
                f"{metrics['Accuracy']:<12.4f} "
                f"{metrics['F1']:<12.4f} "
                f"{metrics['Precision']:<12.4f} "
                f"{metrics['Recall']:<12.4f} "
                f"{metrics['AUC-ROC']:<12.4f}\n")
    
    f.write("\n" + "="*80 + "\n\n")
    
    f.write("FEATURE IMPORTANCE SUMMARY:\n")
    f.write("-" * 50 + "\n")
    for model_name, imp_series in feature_importance_dict.items():
        f.write(f"\n{model_name} - Top 10 Features:\n")
        for feat, imp in imp_series.nlargest(10).items():
            f.write(f"  {feat}: {imp:.6f}\n")
    
    f.write(f"\nDetailed results saved to: {results_filename}\n")

print(f"\nComprehensive results saved to: {results_filename}")

# Also save results in CSV format for easy analysis
csv_filename = f"feature_selection_metrics_{timestamp}.csv"
metrics_data = []

for clf_name, results in all_classifier_results.items():
    row = {
        'Classifier': clf_name,
        'CV_Folds': results['total_cv_folds'],
        'Accuracy_Mean': results['avg_metrics']['Accuracy'],
        'Accuracy_Std': results['std_metrics']['Accuracy'],
        'F1_Mean': results['avg_metrics']['F1'],
        'F1_Std': results['std_metrics']['F1'],
        'Precision_Mean': results['avg_metrics']['Precision'],
        'Precision_Std': results['std_metrics']['Precision'],
        'Recall_Mean': results['avg_metrics']['Recall'],
        'Recall_Std': results['std_metrics']['Recall'],
        'AUC_ROC_Mean': results['avg_metrics']['AUC-ROC'],
        'AUC_ROC_Std': results['std_metrics']['AUC-ROC'],
        'MCC_Mean': results['avg_metrics']['MCC'],
        'MCC_Std': results['std_metrics']['MCC'],
        'Top_Feature_1': results['top_features'].index[0] if len(results['top_features']) > 0 else '',
        'Top_Feature_2': results['top_features'].index[1] if len(results['top_features']) > 1 else '',
        'Top_Feature_3': results['top_features'].index[2] if len(results['top_features']) > 2 else '',
        'True_Negative': results['confusion_matrix'][0,0],
        'False_Positive': results['confusion_matrix'][0,1],
        'False_Negative': results['confusion_matrix'][1,0],
        'True_Positive': results['confusion_matrix'][1,1]
    }
    metrics_data.append(row)

# Save to CSV
metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_csv(csv_filename, index=False)
print(f"Metrics summary saved to CSV: {csv_filename}")

print("=" * 80)



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