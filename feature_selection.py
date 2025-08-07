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
from scipy.stats import wilcoxon, mannwhitneyu, kruskal, chi2_contingency, ks_2samp
import warnings
import datetime
import os

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

dataset_file = 'peptide_baza_balanced.csv'

def log_configuration_info():
    """Log detailed configuration information for reproducibility"""
    print("=" * 80)
    print("FEATURE SELECTION AND CLASSIFICATION EXPERIMENT")
    print("=" * 80)
    print(f"Experiment started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {os.getcwd()}")
    print()
    
    # Dataset information
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

def statistical_feature_selection(X, y, method='mann_whitney', top_k=10, alpha=0.05):
    """
    Perform statistical feature selection using various statistical tests
    
    Parameters:
    X (pd.DataFrame): Feature matrix
    y (pd.Series): Binary target variable
    method (str): 'mann_whitney', 'wilcoxon', 'kruskal', 'chi2', or 'ks_2samp'
    top_k (int): Number of top features to select
    alpha (float): Significance level
    
    Returns:
    tuple: (selected_features, results_dict)
    """
    
    print(f"Performing {method} statistical feature selection...")
    
    if len(np.unique(y)) != 2:
        raise ValueError("Statistical tests require binary classification")
    
    results = {}
    unique_classes = np.unique(y)
    
    if method == 'mann_whitney':
        # Mann-Whitney U test - better for independent samples
        class_0_indices = y[y == unique_classes[0]].index
        class_1_indices = y[y == unique_classes[1]].index
        
        for feature in X.columns:
            values_0 = X.loc[class_0_indices, feature].values
            values_1 = X.loc[class_1_indices, feature].values
            
            try:
                statistic, p_value = mannwhitneyu(values_0, values_1, alternative='two-sided')
                
                # Calculate effect size (rank-biserial correlation)
                n1, n2 = len(values_0), len(values_1)
                effect_size = 1 - (2 * statistic) / (n1 * n2)
                
                results[feature] = {
                    'p_value': p_value,
                    'statistic': statistic,
                    'significant': p_value < alpha,
                    'effect_size': abs(effect_size)
                }
            except:
                results[feature] = {'p_value': 1.0, 'statistic': 0, 'significant': False, 'effect_size': 0}
    
    elif method == 'wilcoxon':
        # Wilcoxon signed rank test - adapted for independent samples
        class_0_indices = y[y == unique_classes[0]].index
        class_1_indices = y[y == unique_classes[1]].index
        
        for feature in X.columns:
            values_0 = X.loc[class_0_indices, feature].values
            values_1 = X.loc[class_1_indices, feature].values
            
            # Create paired samples by taking equal sample sizes
            min_size = min(len(values_0), len(values_1))
            
            if min_size < 6:  # Need minimum samples for Wilcoxon test
                results[feature] = {'p_value': 1.0, 'statistic': 0, 'significant': False, 'effect_size': 0}
                continue
            
            # Random sampling to create pairs (reproducible)
            np.random.seed(42)
            if len(values_0) > min_size:
                idx = np.random.choice(len(values_0), min_size, replace=False)
                values_0 = values_0[idx]
            if len(values_1) > min_size:
                idx = np.random.choice(len(values_1), min_size, replace=False)
                values_1 = values_1[idx]
            
            # Calculate differences
            differences = values_1 - values_0
            differences = differences[differences != 0]
            
            if len(differences) == 0:
                results[feature] = {'p_value': 1.0, 'statistic': 0, 'significant': False, 'effect_size': 0}
                continue
            
            try:
                statistic, p_value = wilcoxon(differences, alternative='two-sided')
                effect_size = np.abs(np.median(differences))
                
                results[feature] = {
                    'p_value': p_value,
                    'statistic': statistic,
                    'significant': p_value < alpha,
                    'effect_size': effect_size
                }
            except:
                results[feature] = {'p_value': 1.0, 'statistic': 0, 'significant': False, 'effect_size': 0}
    
    elif method == 'kruskal':
        # Kruskal-Wallis H test - non-parametric version of ANOVA
        class_0_indices = y[y == unique_classes[0]].index
        class_1_indices = y[y == unique_classes[1]].index
        
        for feature in X.columns:
            values_0 = X.loc[class_0_indices, feature].values
            values_1 = X.loc[class_1_indices, feature].values
            
            try:
                statistic, p_value = kruskal(values_0, values_1)
                
                # Calculate effect size (eta-squared approximation)
                n_total = len(values_0) + len(values_1)
                effect_size = (statistic - 1) / (n_total - 1)
                
                results[feature] = {
                    'p_value': p_value,
                    'statistic': statistic,
                    'significant': p_value < alpha,
                    'effect_size': abs(effect_size)
                }
            except:
                results[feature] = {'p_value': 1.0, 'statistic': 0, 'significant': False, 'effect_size': 0}
    
    elif method == 'chi2':
        # Chi-square test for categorical features (discretized continuous features)
        for feature in X.columns:
            try:
                # Discretize continuous features into bins
                feature_values = X[feature].values
                
                # Create bins based on quartiles
                quartiles = np.percentile(feature_values, [25, 50, 75])
                discretized = np.digitize(feature_values, quartiles)
                
                # Create contingency table
                contingency_table = pd.crosstab(discretized, y)
                
                # Ensure minimum expected frequency
                if contingency_table.min().min() < 5:
                    # Use fewer bins if expected frequencies are too low
                    median_val = np.median(feature_values)
                    discretized = (feature_values > median_val).astype(int)
                    contingency_table = pd.crosstab(discretized, y)
                
                if contingency_table.min().min() >= 1:  # At least 1 in each cell
                    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
                    
                    # Calculate effect size (Cramér's V)
                    n = contingency_table.sum().sum()
                    effect_size = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
                    
                    results[feature] = {
                        'p_value': p_value,
                        'statistic': chi2_stat,
                        'significant': p_value < alpha,
                        'effect_size': effect_size
                    }
                else:
                    results[feature] = {'p_value': 1.0, 'statistic': 0, 'significant': False, 'effect_size': 0}
                    
            except:
                results[feature] = {'p_value': 1.0, 'statistic': 0, 'significant': False, 'effect_size': 0}
    
    elif method == 'ks_2samp':
        # Kolmogorov-Smirnov two-sample test
        class_0_indices = y[y == unique_classes[0]].index
        class_1_indices = y[y == unique_classes[1]].index
        
        for feature in X.columns:
            values_0 = X.loc[class_0_indices, feature].values
            values_1 = X.loc[class_1_indices, feature].values
            
            try:
                statistic, p_value = ks_2samp(values_0, values_1)
                
                # KS statistic is already a measure of effect size (max difference between CDFs)
                effect_size = statistic
                
                results[feature] = {
                    'p_value': p_value,
                    'statistic': statistic,
                    'significant': p_value < alpha,
                    'effect_size': effect_size
                }
            except:
                results[feature] = {'p_value': 1.0, 'statistic': 0, 'significant': False, 'effect_size': 0}
    
    # Select top features based on significance and p-value
    feature_scores = [(feature, data['p_value'], data['significant']) 
                     for feature, data in results.items()]
    feature_scores.sort(key=lambda x: (not x[2], x[1]))  # Sort by significance first, then p-value
    
    selected_features = [feature for feature, _, _ in feature_scores[:top_k]]
    
    # Log results
    significant_count = sum(1 for data in results.values() if data['significant'])
    print(f"  Significant features found: {significant_count}/{len(results)}")
    print(f"  Selected top {len(selected_features)} features")
    
    if len(selected_features) > 0:
        print(f"  Top 3 features: {', '.join(selected_features[:3])}")
    
    return selected_features, results

# Log configuration and load data
data = log_configuration_info()
if data is None:
    exit(1)

# Load and prepare data with logging
columns_to_drop = ["id", "peptide_seq", "targetcol", "hydrophobic_cornette", "synthesis_flag"]
X, y = log_preprocessing_info(data, columns_to_drop)

# STATISTICAL FEATURE SELECTION
print()
print("STATISTICAL FEATURE SELECTION:")
print("-" * 50)

# Perform all statistical tests feature selection
mann_whitney_features, mw_results = statistical_feature_selection(
    X, y, method='mann_whitney', top_k=15, alpha=0.05
)

wilcoxon_features, w_results = statistical_feature_selection(
    X, y, method='wilcoxon', top_k=15, alpha=0.05
)

kruskal_features, k_results = statistical_feature_selection(
    X, y, method='kruskal', top_k=15, alpha=0.05
)

chi2_features, chi_results = statistical_feature_selection(
    X, y, method='chi2', top_k=15, alpha=0.05
)

ks_features, ks_results = statistical_feature_selection(
    X, y, method='ks_2samp', top_k=15, alpha=0.05
)

print(f"\nStatistical Feature Selection Results:")
print(f"Mann-Whitney selected: {len(mann_whitney_features)} features")
print(f"Wilcoxon selected: {len(wilcoxon_features)} features")
print(f"Kruskal-Wallis selected: {len(kruskal_features)} features")
print(f"Chi-square selected: {len(chi2_features)} features")
print(f"Kolmogorov-Smirnov selected: {len(ks_features)} features")

# Find common features between all statistical methods
all_statistical_features = [mann_whitney_features, wilcoxon_features, kruskal_features, 
                           chi2_features, ks_features]
common_all_methods = set(mann_whitney_features)
for features in all_statistical_features[1:]:
    common_all_methods = common_all_methods.intersection(set(features))

print(f"Common features across all statistical methods: {len(common_all_methods)}")
if len(common_all_methods) > 0:
    print(f"Common features: {list(common_all_methods)[:10]}")  # Show top 10

# Find common features between traditional methods (Mann-Whitney + Wilcoxon)
common_statistical_features = list(set(mann_whitney_features) & set(wilcoxon_features))
print(f"Common features between Mann-Whitney and Wilcoxon: {len(common_statistical_features)}")

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
    },
    # Statistical feature selection classifiers
    "Random Forest (Mann-Whitney)": {
        'model': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        'selector': None,  # Features pre-selected
        'features': mann_whitney_features
    },
    "Random Forest (Wilcoxon)": {
        'model': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        'selector': None,  # Features pre-selected
        'features': wilcoxon_features
    },
    "Random Forest (Kruskal)": {
        'model': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        'selector': None,  # Features pre-selected
        'features': kruskal_features
    },
    "Random Forest (Chi-square)": {
        'model': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        'selector': None,  # Features pre-selected
        'features': chi2_features
    },
    "Random Forest (Kolmogorov-Smirnov)": {
        'model': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        'selector': None,  # Features pre-selected
        'features': ks_features
    },
    "Logistic Regression (Mann-Whitney)": {
        'model': make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight='balanced',
                               solver='saga', penalty='l1', C=0.1,
                               random_state=42, n_jobs=-1)
        ),
        'selector': None,  # Features pre-selected
        'features': mann_whitney_features
    },
    "Logistic Regression (Wilcoxon)": {
        'model': make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight='balanced',
                               solver='saga', penalty='l1', C=0.1,
                               random_state=42, n_jobs=-1)
        ),
        'selector': None,  # Features pre-selected
        'features': wilcoxon_features
    },
    "Logistic Regression (Kruskal)": {
        'model': make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight='balanced',
                               solver='saga', penalty='l1', C=0.1,
                               random_state=42, n_jobs=-1)
        ),
        'selector': None,  # Features pre-selected
        'features': kruskal_features
    },
    "Logistic Regression (Chi-square)": {
        'model': make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight='balanced',
                               solver='saga', penalty='l1', C=0.1,
                               random_state=42, n_jobs=-1)
        ),
        'selector': None,  # Features pre-selected
        'features': chi2_features
    },
    "Logistic Regression (Kolmogorov-Smirnov)": {
        'model': make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight='balanced',
                               solver='saga', penalty='l1', C=0.1,
                               random_state=42, n_jobs=-1)
        ),
        'selector': None,  # Features pre-selected
        'features': ks_features
    },
    "SVM (Mann-Whitney)": {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel='rbf', C=0.5, probability=True, random_state=42))
        ]),
        'selector': None,  # Features pre-selected
        'features': mann_whitney_features
    },
    "SVM (Wilcoxon)": {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel='rbf', C=0.5, probability=True, random_state=42))
        ]),
        'selector': None,  # Features pre-selected
        'features': wilcoxon_features
    },
    "SVM (Kruskal)": {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel='rbf', C=0.5, probability=True, random_state=42))
        ]),
        'selector': None,  # Features pre-selected
        'features': kruskal_features
    },
    "SVM (Chi-square)": {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel='rbf', C=0.5, probability=True, random_state=42))
        ]),
        'selector': None,  # Features pre-selected
        'features': chi2_features
    },
    "SVM (Kolmogorov-Smirnov)": {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel='rbf', C=0.5, probability=True, random_state=42))
        ]),
        'selector': None,  # Features pre-selected
        'features': ks_features
    },
    # K-Neighbors with statistical feature selection
    "K-Neighbors (Mann-Whitney)": {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=15, algorithm='kd_tree'))
        ]),
        'selector': None,  # Features pre-selected
        'features': mann_whitney_features
    },
    "K-Neighbors (Wilcoxon)": {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=15, algorithm='kd_tree'))
        ]),
        'selector': None,  # Features pre-selected
        'features': wilcoxon_features
    },
    "K-Neighbors (Kruskal)": {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=15, algorithm='kd_tree'))
        ]),
        'selector': None,  # Features pre-selected
        'features': kruskal_features
    },
    "K-Neighbors (Chi-square)": {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=15, algorithm='kd_tree'))
        ]),
        'selector': None,  # Features pre-selected
        'features': chi2_features
    },
    "K-Neighbors (Kolmogorov-Smirnov)": {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=15, algorithm='kd_tree'))
        ]),
        'selector': None,  # Features pre-selected
        'features': ks_features
    },
    # Decision Tree with statistical feature selection
    "Decision Tree (Mann-Whitney)": {
        'model': DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42),
        'selector': None,  # Features pre-selected
        'features': mann_whitney_features
    },
    "Decision Tree (Wilcoxon)": {
        'model': DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42),
        'selector': None,  # Features pre-selected
        'features': wilcoxon_features
    },
    "Decision Tree (Kruskal)": {
        'model': DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42),
        'selector': None,  # Features pre-selected
        'features': kruskal_features
    },
    "Decision Tree (Chi-square)": {
        'model': DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42),
        'selector': None,  # Features pre-selected
        'features': chi2_features
    },
    "Decision Tree (Kolmogorov-Smirnov)": {
        'model': DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42),
        'selector': None,  # Features pre-selected
        'features': ks_features
    },
    # Naive Bayes with statistical feature selection
    "Naive Bayes (Mann-Whitney)": {
        'model': make_pipeline(
            StandardScaler(),
            GaussianNB()
        ),
        'selector': None,  # Features pre-selected
        'features': mann_whitney_features
    },
    "Naive Bayes (Wilcoxon)": {
        'model': make_pipeline(
            StandardScaler(),
            GaussianNB()
        ),
        'selector': None,  # Features pre-selected
        'features': wilcoxon_features
    },
    "Naive Bayes (Kruskal)": {
        'model': make_pipeline(
            StandardScaler(),
            GaussianNB()
        ),
        'selector': None,  # Features pre-selected
        'features': kruskal_features
    },
    "Naive Bayes (Chi-square)": {
        'model': make_pipeline(
            StandardScaler(),
            GaussianNB()
        ),
        'selector': None,  # Features pre-selected
        'features': chi2_features
    },
    "Naive Bayes (Kolmogorov-Smirnov)": {
        'model': make_pipeline(
            StandardScaler(),
            GaussianNB()
        ),
        'selector': None,  # Features pre-selected
        'features': ks_features
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
    "Naive Bayes": pd.Series(0.0, index=X.columns),
    "Random Forest (Mann-Whitney)": pd.Series(0.0, index=X.columns),
    "Random Forest (Wilcoxon)": pd.Series(0.0, index=X.columns),
    "Random Forest (Kruskal)": pd.Series(0.0, index=X.columns),
    "Random Forest (Chi-square)": pd.Series(0.0, index=X.columns),
    "Random Forest (Kolmogorov-Smirnov)": pd.Series(0.0, index=X.columns),
    "Logistic Regression (Mann-Whitney)": pd.Series(0.0, index=X.columns),
    "Logistic Regression (Wilcoxon)": pd.Series(0.0, index=X.columns),
    "Logistic Regression (Kruskal)": pd.Series(0.0, index=X.columns),
    "Logistic Regression (Chi-square)": pd.Series(0.0, index=X.columns),
    "Logistic Regression (Kolmogorov-Smirnov)": pd.Series(0.0, index=X.columns),
    "SVM (Mann-Whitney)": pd.Series(0.0, index=X.columns),
    "SVM (Wilcoxon)": pd.Series(0.0, index=X.columns),
    "SVM (Kruskal)": pd.Series(0.0, index=X.columns),
    "SVM (Chi-square)": pd.Series(0.0, index=X.columns),
    "SVM (Kolmogorov-Smirnov)": pd.Series(0.0, index=X.columns),
    "K-Neighbors (Mann-Whitney)": pd.Series(0.0, index=X.columns),
    "K-Neighbors (Wilcoxon)": pd.Series(0.0, index=X.columns),
    "K-Neighbors (Kruskal)": pd.Series(0.0, index=X.columns),
    "K-Neighbors (Chi-square)": pd.Series(0.0, index=X.columns),
    "K-Neighbors (Kolmogorov-Smirnov)": pd.Series(0.0, index=X.columns),
    "Decision Tree (Mann-Whitney)": pd.Series(0.0, index=X.columns),
    "Decision Tree (Wilcoxon)": pd.Series(0.0, index=X.columns),
    "Decision Tree (Kruskal)": pd.Series(0.0, index=X.columns),
    "Decision Tree (Chi-square)": pd.Series(0.0, index=X.columns),
    "Decision Tree (Kolmogorov-Smirnov)": pd.Series(0.0, index=X.columns),
    "Naive Bayes (Mann-Whitney)": pd.Series(0.0, index=X.columns),
    "Naive Bayes (Wilcoxon)": pd.Series(0.0, index=X.columns),
    "Naive Bayes (Kruskal)": pd.Series(0.0, index=X.columns),
    "Naive Bayes (Chi-square)": pd.Series(0.0, index=X.columns),
    "Naive Bayes (Kolmogorov-Smirnov)": pd.Series(0.0, index=X.columns)
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
        
        # Check if features are pre-selected (statistical methods)
        if 'features' in clf_info:
            # Use pre-selected features from statistical tests
            selected_features = clf_info['features']
            X_train_selected = X_train[selected_features]
            X_test_selected = X_test[selected_features]
            
            # Train model with pre-selected features
            model = clf_info['model'].fit(X_train_selected, y_train)
            
            # For feature importance, use model-based importance if available
            if hasattr(model, 'feature_importances_'):
                importances = pd.Series(model.feature_importances_, index=selected_features)
            elif hasattr(model, 'coef_') and hasattr(model.coef_, 'shape'):
                if len(model.coef_.shape) > 1:
                    importances = pd.Series(np.abs(model.coef_[0]), index=selected_features)
                else:
                    importances = pd.Series(np.abs(model.coef_), index=selected_features)
            elif hasattr(model.named_steps, 'logisticregression'):
                # For pipeline with logistic regression
                coefs = model.named_steps['logisticregression'].coef_[0]
                importances = pd.Series(np.abs(coefs), index=selected_features)
            else:
                # Use statistical test p-values as importance (inverse)
                if clf_name.endswith("(Mann-Whitney)"):
                    stat_results = mw_results
                elif clf_name.endswith("(Wilcoxon)"):
                    stat_results = w_results
                elif clf_name.endswith("(Kruskal)"):
                    stat_results = k_results
                elif clf_name.endswith("(Chi-square)"):
                    stat_results = chi_results
                elif clf_name.endswith("(Kolmogorov-Smirnov)"):
                    stat_results = ks_results
                else:
                    stat_results = {}
                
                importances = pd.Series([1.0 / (stat_results.get(feat, {'p_value': 1.0})['p_value'] + 1e-10) 
                                       for feat in selected_features], index=selected_features)
            
            # Update global feature importance
            full_importances = pd.Series(0.0, index=X.columns)
            full_importances[selected_features] = importances.values
            model_feature_importance += full_importances
            
        elif clf_info['selector'] is not None:
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
results_filename = f"feature_selection_results_{dataset_file}_{timestamp}.txt"

with open(results_filename, 'w') as f:
    f.write("PEPTIDE SYNTHESIS PREDICTION - FEATURE SELECTION RESULTS\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Experiment Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Dataset: {dataset_file}\n")
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