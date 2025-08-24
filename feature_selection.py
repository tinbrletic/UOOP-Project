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
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, RFECV
from skrebate import ReliefF
from scipy.stats import wilcoxon, mannwhitneyu, kruskal, chi2_contingency, ks_2samp, friedmanchisquare, norm, rankdata
import itertools
import warnings
import datetime
import os

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# dataset_file = 'peptide_baza_balanced.csv'
dataset_file = 'peptide_baza_formatted.csv'

# --- Integrated selector constructors ---
def make_integrated_selector_logreg_l1(C=0.1, class_weight='balanced', random_state=42, max_iter=5000):
    lr = LogisticRegression(
        penalty='l1', solver='saga', C=C, class_weight=class_weight,
        random_state=random_state, max_iter=max_iter, n_jobs=-1
    )
    return SelectFromModel(estimator=lr, threshold='median')

def make_integrated_selector_rf(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1, threshold='median'):
    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=n_jobs
    )
    return SelectFromModel(estimator=rf, threshold=threshold)

def make_rfecv_logreg_l1(C=0.1, class_weight='balanced', random_state=42, max_iter=5000, step=0.1, cv=5, n_jobs=-1):
    est = LogisticRegression(
        penalty='l1', solver='saga', C=C, class_weight=class_weight,
        random_state=random_state, max_iter=max_iter, n_jobs=n_jobs
    )
    return RFECV(estimator=est, step=step, cv=cv, scoring='f1', n_jobs=n_jobs)


def extract_feature_importance_from_pipeline(model_fitted, X_columns):
    """Extract feature importances from integrated pipelines (SelectFromModel/RFECV/clf)."""
    if hasattr(model_fitted, 'named_steps'):
        steps = model_fitted.named_steps

        # SelectFromModel path
        if 'selector' in steps and isinstance(steps['selector'], SelectFromModel):
            selector = steps['selector']
            if hasattr(selector, 'get_support'):
                support_mask = selector.get_support(indices=False)
                selected_cols = np.array(X_columns)[support_mask]
            else:
                selected_cols = np.array(X_columns)
            est = getattr(selector, 'estimator_', None) or selector.estimator
            if hasattr(est, 'feature_importances_'):
                vals = est.feature_importances_
                return pd.Series({c: v for c, v in zip(selected_cols, vals[:len(selected_cols)])})
            if hasattr(est, 'coef_'):
                coefs = est.coef_[0] if len(est.coef_.shape) > 1 else est.coef_.ravel()
                abs_coefs = np.abs(coefs)
                return pd.Series({c: v for c, v in zip(selected_cols, abs_coefs[:len(selected_cols)])})
            return pd.Series({c: 1.0 for c in selected_cols})

        # RFECV path
        if 'rfecv' in steps and isinstance(steps['rfecv'], RFECV):
            rfecv = steps['rfecv']
            if hasattr(rfecv, 'support_'):
                support_mask = rfecv.support_
                selected_cols = np.array(X_columns)[support_mask]
            else:
                selected_cols = np.array(X_columns)
            est = getattr(rfecv, 'estimator_', None) or rfecv.estimator
            if hasattr(est, 'coef_') and hasattr(rfecv, 'support_'):
                coefs = est.coef_[0] if len(est.coef_.shape) > 1 else est.coef_.ravel()
                if len(coefs) == len(rfecv.support_):
                    vals = np.abs(coefs[rfecv.support_])
                    return pd.Series({c: v for c, v in zip(selected_cols, vals)})
            if hasattr(est, 'feature_importances_'):
                vals = est.feature_importances_
                return pd.Series({c: v for c, v in zip(selected_cols, vals[:len(selected_cols)])})
            return pd.Series({c: 1.0 for c in selected_cols})

        # Direct classifier importances (if no selector step)
        if 'clf' in steps:
            clf = steps['clf']
            if hasattr(clf, 'feature_importances_'):
                return pd.Series(clf.feature_importances_, index=X_columns)
            if hasattr(clf, 'coef_'):
                coefs = clf.coef_[0] if len(clf.coef_.shape) > 1 else clf.coef_.ravel()
                return pd.Series(np.abs(coefs), index=X_columns)

    return pd.Series(index=X_columns, data=0.0)


# --- Analysis helpers for logging and reporting ---
def analyze_feature_overlap(feature_lists, method_names):
    # Create DataFrame with boolean indicators
    all_features = set()
    for features in feature_lists:
        all_features.update(features)

    all_features_sorted = sorted(all_features)
    overlap_data = {}
    for method, features in zip(method_names, feature_lists):
        overlap_data[method] = [1 if feat in features else 0 for feat in all_features_sorted]

    overlap_df = pd.DataFrame(overlap_data, index=all_features_sorted)
    overlap_df['Total_Methods'] = overlap_df.sum(axis=1)

    # Logging
    print("\nFEATURE OVERLAP ANALYSIS:")
    print("-" * 50)
    method_to_set = {m: set(fls) for m, fls in zip(method_names, feature_lists)}
    for n_methods in range(len(method_names), 0, -1):
        features_in_n = overlap_df[overlap_df['Total_Methods'] == n_methods]
        if len(features_in_n) > 0:
            print(f"Features selected by {n_methods} methods ({len(features_in_n)}):")
            for feat in features_in_n.index:
                methods = [m for m in method_names if feat in method_to_set[m]]
                print(f"  {feat:30} -> {', '.join(methods)}")

    return overlap_df


def analyze_integrated_stability(features_per_fold, model_name):
    print(f"\n{model_name} - FEATURE SELECTION STABILITY:")
    print("-" * 60)

    feature_freq = {}
    total_folds = len(features_per_fold)
    for fold_features in features_per_fold:
        for feature in fold_features:
            feature_freq[feature] = feature_freq.get(feature, 0) + 1

    sorted_features = sorted(feature_freq.items(), key=lambda x: x[1], reverse=True)
    print(f"Most frequently selected features (out of {total_folds} folds):")
    for feature, freq in sorted_features[:15]:
        percentage = (freq / max(total_folds, 1)) * 100
        print(f"  {feature:30} {freq:3d}/{total_folds} ({percentage:5.1f}%)")

    if len(sorted_features) > 0:
        avg_selection_freq = np.mean(list(feature_freq.values()))
        stability_score = len([f for f, freq in feature_freq.items() if freq >= total_folds * 0.5]) / max(len(feature_freq), 1)
        print(f"\nStability metrics:")
        print(f"  Average selection frequency: {avg_selection_freq:.1f}/{total_folds}")
        print(f"  Features selected in >50% folds: {stability_score:.2f}")

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

# --- Multiple testing correction helpers ---
def p_adjust(pvals, method='holm'):
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.size
    if n == 0:
        return pvals
    m = method.lower()
    if m in ['bh', 'fdr_bh', 'benjamini-hochberg', 'fdr']:
        order = np.argsort(pvals)
        ranked = pvals[order]
        adj = np.empty(n, dtype=float)
        cummin_input = (n / (np.arange(n, 0, -1))) * ranked[::-1]
        adj_rev = np.minimum.accumulate(cummin_input)[::-1]
        adj[order] = np.minimum(adj_rev, 1.0)
        return adj
    elif m in ['holm', 'holm-bonferroni']:
        order = np.argsort(pvals)
        ranked = pvals[order]
        adj = np.empty(n, dtype=float)
        adj_vals = ranked * (n - np.arange(n))
        adj_vals = np.maximum.accumulate(adj_vals)
        adj[order] = np.minimum(adj_vals, 1.0)
        return adj
    else:
        # Bonferroni fallback
        return np.minimum(pvals * n, 1.0)


def collect_metric_matrix(all_results, metric_name='AUC-ROC'):
    model_names = []
    fold_vectors = []
    # First, gather all vectors
    for mname, res in all_results.items():
        if 'fold_metrics' not in res or metric_name not in res['fold_metrics']:
            continue
        v = list(res['fold_metrics'][metric_name])
        model_names.append(mname)
        fold_vectors.append(v)
    if not fold_vectors:
        return pd.DataFrame()
    # Truncate all vectors to the shortest length so columns align fold-wise
    min_len = min(len(v) for v in fold_vectors)
    fold_vectors = [v[:min_len] for v in fold_vectors]
    df = pd.DataFrame({m: fold_vectors[i] for i, m in enumerate(model_names)})
    return df


def run_friedman(metric_df):
    arrays = [metric_df[c].values for c in metric_df.columns]
    stat, p = friedmanchisquare(*arrays)
    ranks_per_fold = metric_df.rank(axis=1, ascending=False, method='average')
    avg_ranks = ranks_per_fold.mean(axis=0).sort_values()
    return {'statistic': float(stat), 'p_value': float(p), 'avg_ranks': avg_ranks.to_dict()}


def posthoc_conover(metric_df, p_adjust_method='holm'):
    import importlib
    try:
        sp = importlib.import_module('scikit_posthocs')
    except Exception as e:
        print(f"Conover not available ({e}). Falling back to Nemenyi.")
        return None
    ranks = metric_df.rank(axis=1, ascending=False, method='average')
    ph = sp.posthoc_conover(ranks, p_adjust=p_adjust_method)
    ph.index = metric_df.columns
    ph.columns = metric_df.columns
    return ph


def posthoc_nemenyi(metric_df, alpha=0.05):
    ranks = metric_df.rank(axis=1, ascending=False, method='average')
    avg_ranks = ranks.mean(axis=0)
    # Try to compute p-values via scikit-posthocs if available
    import importlib
    try:
        sp = importlib.import_module('scikit_posthocs')
        nemenyi = sp.posthoc_nemenyi(metric_df)
        nemenyi.index = metric_df.columns
        nemenyi.columns = metric_df.columns
        return {'avg_ranks': avg_ranks.to_dict(), 'pvals': nemenyi}
    except Exception:
        return {'avg_ranks': avg_ranks.to_dict(), 'pvals': None}


def pairwise_wilcoxon(metric_df, pairs=None, p_adjust_method='holm'):
    cols = list(metric_df.columns)
    if pairs is None:
        pairs = list(itertools.combinations(cols, 2))
    results = []
    n = metric_df.shape[0]
    for a, b in pairs:
        x = metric_df[a].values
        y = metric_df[b].values
        stat, p = wilcoxon(x, y, zero_method='wilcox', alternative='two-sided', mode='auto')
        z = float(np.sign(np.median(x - y)) * norm.isf(p/2.0)) if p > 0 else float('inf')
        r = z / np.sqrt(n)
        results.append({'Model_A': a, 'Model_B': b, 'W': float(stat), 'p_value': float(p), 'z': float(z), 'r_effect': float(r)})
    pvals = [r['p_value'] for r in results]
    adj = p_adjust(np.array(pvals), method=p_adjust_method)
    for i, r in enumerate(results):
        r['p_adj'] = float(adj[i])
    return pd.DataFrame(results)


def run_statistical_evaluation(all_results, metric_name='AUC-ROC', alpha=0.05, p_adjust_method='holm', out_prefix=None):
    if out_prefix is None:
        out_prefix = f"stats_{metric_name.replace('-', '_').replace(' ', '')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    metric_df = collect_metric_matrix(all_results, metric_name=metric_name)
    if metric_df.shape[1] < 3:
        print(f"[WARN] Need at least 3 models for Friedman. Found: {metric_df.shape[1]}")
    metric_df.to_csv(f"{out_prefix}_fold_metrics.csv", index=False)
    print(f"[OK] Saved per-fold {metric_name} matrix to {out_prefix}_fold_metrics.csv")
    fried = run_friedman(metric_df)
    with open(f"{out_prefix}_friedman.json", "w") as f:
        import json
        json.dump(fried, f, indent=2)
    print(f"[FRIEDMAN] statistic={fried['statistic']:.4f}, p={fried['p_value']:.6f}")
    print("[FRIEDMAN] Average ranks (lower is better):")
    for m, r in sorted(fried['avg_ranks'].items(), key=lambda x: x[1]):
        print(f"  {m:40} {r:.3f}")
    conover = posthoc_conover(metric_df, p_adjust_method=p_adjust_method)
    if conover is not None:
        conover.to_csv(f"{out_prefix}_posthoc_conover_{p_adjust_method}.csv")
        print(f"[POSTHOC] Conover + {p_adjust_method} saved to {out_prefix}_posthoc_conover_{p_adjust_method}.csv")
    nemenyi = posthoc_nemenyi(metric_df)
    if isinstance(nemenyi, dict):
        pd.Series(nemenyi['avg_ranks']).sort_values().to_csv(f"{out_prefix}_nemenyi_avg_ranks.csv", header=['avg_rank'])
        if nemenyi['pvals'] is not None:
            nemenyi['pvals'].to_csv(f"{out_prefix}_posthoc_nemenyi.csv")
            print(f"[POSTHOC] Nemenyi saved to {out_prefix}_posthoc_nemenyi.csv")
        else:
            print("[POSTHOC] Nemenyi p-values unavailable (scikit-posthocs not installed). Saved avg ranks.")
    pw = pairwise_wilcoxon(metric_df, pairs=None, p_adjust_method=p_adjust_method)
    pw.to_csv(f"{out_prefix}_pairwise_wilcoxon_{p_adjust_method}.csv", index=False)
    print(f"[WILCOXON] Pairwise results saved to {out_prefix}_pairwise_wilcoxon_{p_adjust_method}.csv")
    best_by_mean = metric_df.mean(axis=0).sort_values(ascending=False)
    print("\n[SUMMARY] Mean AUC-ROC by model:")
    for m, v in best_by_mean.items():
        print(f"  {m:40} {v:.4f}")

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
                # Discretize continuous features into bins (quartiles)
                feature_values = X[feature].values
                quartiles = np.percentile(feature_values, [25, 50, 75])
                discretized = np.digitize(feature_values, quartiles)

                # Create contingency table and ensure both target classes are present as columns
                contingency_table = pd.crosstab(discretized, y)
                contingency_table = contingency_table.reindex(columns=unique_classes, fill_value=0)

                # If low expected counts, fallback to median split (2 bins)
                if contingency_table.min().min() < 5:
                    median_val = np.median(feature_values)
                    discretized = (feature_values > median_val).astype(int)
                    contingency_table = pd.crosstab(discretized, y)
                    contingency_table = contingency_table.reindex(columns=unique_classes, fill_value=0)

                # Require at least a 2x2 table with no empty cells
                r, c = contingency_table.shape
                if r >= 2 and c >= 2 and contingency_table.min().min() >= 1:
                    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

                    # Calculate effect size (Cramér's V) safely
                    n = contingency_table.to_numpy().sum()
                    denom = n * (min(r, c) - 1)
                    if denom > 0:
                        effect_size = float(np.sqrt(chi2_stat / denom))
                    else:
                        effect_size = 0.0

                    results[feature] = {
                        'p_value': float(p_value) if np.isfinite(p_value) else 1.0,
                        'statistic': float(chi2_stat) if np.isfinite(chi2_stat) else 0.0,
                        'significant': np.isfinite(p_value) and (p_value < alpha),
                        'effect_size': float(effect_size)
                    }
                else:
                    # Not enough structure to run a valid chi-square
                    results[feature] = {'p_value': 1.0, 'statistic': 0.0, 'significant': False, 'effect_size': 0.0}

            except Exception:
                results[feature] = {'p_value': 1.0, 'statistic': 0.0, 'significant': False, 'effect_size': 0.0}
    
    elif method == 'ks_2samp':
        # Kolmogorov-Smirnov two-sample test
        class_0_indices = y[y == unique_classes[0]].index
        class_1_indices = y[y == unique_classes[1]].index
        
        for feature in X.columns:
            values_0 = X.loc[class_0_indices, feature].values
            values_1 = X.loc[class_1_indices, feature].values
            
            # Clean NaNs/Infs to ensure valid inputs
            values_0 = values_0[np.isfinite(values_0)]
            values_1 = values_1[np.isfinite(values_1)]
            
            # Require minimum samples in both groups
            if len(values_0) < 2 or len(values_1) < 2:
                results[feature] = {'p_value': 1.0, 'statistic': 0, 'significant': False, 'effect_size': 0}
                continue
            
            try:
                # Explicitly use asymptotic method to avoid exact method warnings with ties/large n
                statistic, p_value = ks_2samp(values_0, values_1, alternative='two-sided', method='asymp')
                
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

# Detailed lists of selected features for each statistical method
print(f"\nMANN_WHITNEY SELECTED FEATURES ({len(mann_whitney_features)}):")
for i, feature in enumerate(mann_whitney_features, 1):
    p_val = mw_results[feature]['p_value']
    print(f"  {i:2d}. {feature:30} (p={p_val:.4f})")

print(f"\nWILCOXON SELECTED FEATURES ({len(wilcoxon_features)}):")
for i, feature in enumerate(wilcoxon_features, 1):
    p_val = w_results[feature]['p_value']
    print(f"  {i:2d}. {feature:30} (p={p_val:.4f})")

print(f"\nKRUSKAL SELECTED FEATURES ({len(kruskal_features)}):")
for i, feature in enumerate(kruskal_features, 1):
    p_val = k_results[feature]['p_value']
    print(f"  {i:2d}. {feature:30} (p={p_val:.4f})")

print(f"\nCHI2 SELECTED FEATURES ({len(chi2_features)}):")
for i, feature in enumerate(chi2_features, 1):
    p_val = chi_results[feature]['p_value']
    print(f"  {i:2d}. {feature:30} (p={p_val:.4f})")

print(f"\nKS SELECTED FEATURES ({len(ks_features)}):")
for i, feature in enumerate(ks_features, 1):
    p_val = ks_results[feature]['p_value']
    print(f"  {i:2d}. {feature:30} (p={p_val:.4f})")

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

# Overlap analysis between statistical methods
feature_lists = [mann_whitney_features, wilcoxon_features, kruskal_features, chi2_features, ks_features]
method_names = ['Mann-Whitney', 'Wilcoxon', 'Kruskal', 'Chi-square', 'KS']
overlap_df = analyze_feature_overlap(feature_lists, method_names)

# Prepare features summary data (will append integrated later)
features_summary_data = []
for method, features in zip(method_names, feature_lists):
    for i, feature in enumerate(features):
        features_summary_data.append({
            'Selection_Method': method,
            'Feature_Name': feature,
            'Rank': i + 1,
            'Selection_Type': 'Statistical'
        })

# Baseline full feature list
all_features = X.columns.tolist()

# Define classifiers including baseline (All) and statistical-selection variants
classifiers = {
    # Baseline (All features)
    "Random Forest (All)": {
        'model': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        'selector': None,
        'features': all_features
    },
    "Logistic Regression (All)": {
        'model': make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight='balanced',
                               solver='saga', penalty='l1', C=0.1,
                               random_state=42, n_jobs=-1)
        ),
        'selector': None,
        'features': all_features
    },
    "SVM (All)": {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel='rbf', C=0.5, probability=True, random_state=42))
        ]),
        'selector': None,
        'features': all_features
    },
    "K-Neighbors (All)": {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=15, algorithm='kd_tree'))
        ]),
        'selector': None,
        'features': all_features
    },
    "Decision Tree (All)": {
        'model': DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42),
        'selector': None,
        'features': all_features
    },
    "Naive Bayes (All)": {
        'model': make_pipeline(
            StandardScaler(),
            GaussianNB()
        ),
        'selector': None,
        'features': all_features
    },
    # Random Forest statistical subsets
    "Random Forest (Mann-Whitney)": {
        'model': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        'selector': None,
        'features': mann_whitney_features
    },
    "Random Forest (Wilcoxon)": {
        'model': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        'selector': None,
        'features': wilcoxon_features
    },
    "Random Forest (Kruskal)": {
        'model': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        'selector': None,
        'features': kruskal_features
    },
    "Random Forest (Chi-square)": {
        'model': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        'selector': None,
        'features': chi2_features
    },
    "Random Forest (Kolmogorov-Smirnov)": {
        'model': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        'selector': None,
        'features': ks_features
    },
    # Logistic Regression statistical subsets
    "Logistic Regression (Mann-Whitney)": {
        'model': make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight='balanced', solver='saga', penalty='l1', C=0.1, random_state=42, n_jobs=-1)
        ),
        'selector': None,
        'features': mann_whitney_features
    },
    "Logistic Regression (Wilcoxon)": {
        'model': make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight='balanced', solver='saga', penalty='l1', C=0.1, random_state=42, n_jobs=-1)
        ),
        'selector': None,
        'features': wilcoxon_features
    },
    "Logistic Regression (Kruskal)": {
        'model': make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight='balanced', solver='saga', penalty='l1', C=0.1, random_state=42, n_jobs=-1)
        ),
        'selector': None,
        'features': kruskal_features
    },
    "Logistic Regression (Chi-square)": {
        'model': make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight='balanced', solver='saga', penalty='l1', C=0.1, random_state=42, n_jobs=-1)
        ),
        'selector': None,
        'features': chi2_features
    },
    "Logistic Regression (Kolmogorov-Smirnov)": {
        'model': make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight='balanced', solver='saga', penalty='l1', C=0.1, random_state=42, n_jobs=-1)
        ),
        'selector': None,
        'features': ks_features
    },
    # SVM statistical subsets
    "SVM (Mann-Whitney)": {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel='rbf', C=0.5, probability=True, random_state=42))
        ]),
        'selector': None,
        'features': mann_whitney_features
    },
    "SVM (Wilcoxon)": {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel='rbf', C=0.5, probability=True, random_state=42))
        ]),
        'selector': None,
        'features': wilcoxon_features
    },
    "SVM (Kruskal)": {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel='rbf', C=0.5, probability=True, random_state=42))
        ]),
        'selector': None,
        'features': kruskal_features
    },
    "SVM (Chi-square)": {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel='rbf', C=0.5, probability=True, random_state=42))
        ]),
        'selector': None,
        'features': chi2_features
    },
    "SVM (Kolmogorov-Smirnov)": {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel='rbf', C=0.5, probability=True, random_state=42))
        ]),
        'selector': None,
        'features': ks_features
    },
    # K-Neighbors statistical subsets
    "K-Neighbors (Mann-Whitney)": {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=15, algorithm='kd_tree'))
        ]),
        'selector': None,
        'features': mann_whitney_features
    },
    "K-Neighbors (Wilcoxon)": {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=15, algorithm='kd_tree'))
        ]),
        'selector': None,
        'features': wilcoxon_features
    },
    "K-Neighbors (Kruskal)": {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=15, algorithm='kd_tree'))
        ]),
        'selector': None,
        'features': kruskal_features
    },
    "K-Neighbors (Chi-square)": {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=15, algorithm='kd_tree'))
        ]),
        'selector': None,
        'features': chi2_features
    },
    "K-Neighbors (Kolmogorov-Smirnov)": {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=15, algorithm='kd_tree'))
        ]),
        'selector': None,
        'features': ks_features
    },
    # Decision Tree statistical subsets
    "Decision Tree (Mann-Whitney)": {
        'model': DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42),
        'selector': None,
        'features': mann_whitney_features
    },
    "Decision Tree (Wilcoxon)": {
        'model': DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42),
        'selector': None,
        'features': wilcoxon_features
    },
    "Decision Tree (Kruskal)": {
        'model': DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42),
        'selector': None,
        'features': kruskal_features
    },
    "Decision Tree (Chi-square)": {
        'model': DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42),
        'selector': None,
        'features': chi2_features
    },
    "Decision Tree (Kolmogorov-Smirnov)": {
        'model': DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42),
        'selector': None,
        'features': ks_features
    },
    # Naive Bayes statistical subsets
    "Naive Bayes (Mann-Whitney)": {
        'model': make_pipeline(StandardScaler(), GaussianNB()),
        'selector': None,
        'features': mann_whitney_features
    },
    "Naive Bayes (Wilcoxon)": {
        'model': make_pipeline(StandardScaler(), GaussianNB()),
        'selector': None,
        'features': wilcoxon_features
    },
    "Naive Bayes (Kruskal)": {
        'model': make_pipeline(StandardScaler(), GaussianNB()),
        'selector': None,
        'features': kruskal_features
    },
    "Naive Bayes (Chi-square)": {
        'model': make_pipeline(StandardScaler(), GaussianNB()),
        'selector': None,
        'features': chi2_features
    },
    "Naive Bayes (Kolmogorov-Smirnov)": {
        'model': make_pipeline(StandardScaler(), GaussianNB()),
        'selector': None,
        'features': ks_features
    },
    # --- Integrated selectors inside pipelines ---
    # L1-LR selector -> LR
    "Integrated-SelectFromModel (L1-LR selector) + LR": {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('selector', make_integrated_selector_logreg_l1(C=0.1)),
            ('clf', LogisticRegression(
                penalty='l2', solver='lbfgs', max_iter=5000, class_weight='balanced', n_jobs=-1, random_state=42
            ))
        ]),
        'selector': 'integrated',
        'features': None
    },
    # L1-LR selector -> SVM
    "Integrated-SelectFromModel (L1-LR selector) + SVM": {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('selector', make_integrated_selector_logreg_l1(C=0.1)),
            ('clf', SVC(kernel='rbf', C=0.5, probability=True, random_state=42))
        ]),
        'selector': 'integrated',
        'features': None
    },
    # L1-LR selector -> KNN
    "Integrated-SelectFromModel (L1-LR selector) + KNN": {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('selector', make_integrated_selector_logreg_l1(C=0.1)),
            ('clf', KNeighborsClassifier(n_neighbors=15, algorithm='kd_tree'))
        ]),
        'selector': 'integrated',
        'features': None
    },
    # L1-LR selector -> Naive Bayes
    "Integrated-SelectFromModel (L1-LR selector) + Naive Bayes": {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('selector', make_integrated_selector_logreg_l1(C=0.1)),
            ('clf', GaussianNB())
        ]),
        'selector': 'integrated',
        'features': None
    },
    # RF selector -> RF
    "Integrated-SelectFromModel (RF selector) + RF": {
        'model': Pipeline([
            ('selector', make_integrated_selector_rf(n_estimators=300, threshold='median')),
            ('clf', RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1))
        ]),
        'selector': 'integrated',
        'features': None
    },
    # RF selector -> LR
    "Integrated-SelectFromModel (RF selector) + LR": {
        'model': Pipeline([
            ('selector', make_integrated_selector_rf(n_estimators=300, threshold='median')),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                penalty='l2', solver='lbfgs', max_iter=5000, class_weight='balanced', n_jobs=-1, random_state=42
            ))
        ]),
        'selector': 'integrated',
        'features': None
    },
    # RF selector -> KNN
    "Integrated-SelectFromModel (RF selector) + KNN": {
        'model': Pipeline([
            ('selector', make_integrated_selector_rf(n_estimators=300, threshold='median')),
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=15, algorithm='kd_tree'))
        ]),
        'selector': 'integrated',
        'features': None
    },
    # RFECV (L1-LR) -> LR
    "Integrated-RFECV (L1-LR) + LR": {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('rfecv', make_rfecv_logreg_l1(C=0.1, cv=5)),
            ('clf', LogisticRegression(
                penalty='l2', solver='lbfgs', max_iter=5000, class_weight='balanced', n_jobs=-1, random_state=42
            ))
        ]),
        'selector': 'integrated',
        'features': None
    },
    # RFECV (L1-LR) -> KNN
    "Integrated-RFECV (L1-LR) + KNN": {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('rfecv', make_rfecv_logreg_l1(C=0.1, cv=5)),
            ('clf', KNeighborsClassifier(n_neighbors=15, algorithm='kd_tree'))
        ]),
        'selector': 'integrated',
        'features': None
    },
    # RFECV (L1-LR) -> Naive Bayes
    "Integrated-RFECV (L1-LR) + Naive Bayes": {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('rfecv', make_rfecv_logreg_l1(C=0.1, cv=5)),
            ('clf', GaussianNB())
        ]),
        'selector': 'integrated',
        'features': None
    }
}

# Configure cross-validation
kf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)

# Log cross-validation configuration
log_cv_configuration(kf)

# Log classifier configurations
log_classifier_configuration(classifiers)

# Global feature importance dictionary (baseline + statistical subsets)
feature_importance_dict = {name: pd.Series(0.0, index=X.columns) for name in classifiers.keys()}

# Initialize results dictionary to store all classifier performance
all_classifier_results = {}

for clf_name, clf_info in classifiers.items():
    print(f"\n{'='*40}\nEvaluating {clf_name}\n{'='*40}")
    
    metrics_history = {metric: [] for metric in ['Accuracy', 'F1', 'AUC-ROC', 'Precision', 'Recall', 'MCC']}
    # Accumulate per-fold feature importance for this classifier.
    model_feature_importance = pd.Series(0.0, index=X.columns)
    confusion_matrices = []
    # Track selected features per fold for integrated selectors
    selected_features_per_fold = [] if clf_info.get('selector', None) == 'integrated' else None
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = clf_info['model']

        if clf_info.get('selector', None) == 'integrated':
            # Fit entire pipeline on train and predict on test
            model_fitted = model.fit(X_train, y_train)
            y_pred = model_fitted.predict(X_test)
            if hasattr(model_fitted, 'predict_proba'):
                y_proba = model_fitted.predict_proba(X_test)[:, 1]
            elif hasattr(model_fitted, 'decision_function'):
                y_proba = model_fitted.decision_function(X_test)
            else:
                y_proba = (y_pred == 1).astype(float)

            # Extract importances from pipeline
            importances_series = extract_feature_importance_from_pipeline(model_fitted, X.columns)
            full_importances = pd.Series(0.0, index=X.columns)
            if len(importances_series) > 0:
                full_importances.loc[importances_series.index] = importances_series.values
            model_feature_importance += full_importances

            # Log selected feature count (if available)
            n_selected = None
            if hasattr(model_fitted, 'named_steps'):
                steps = model_fitted.named_steps
                if 'selector' in steps and hasattr(steps['selector'], 'get_support'):
                    n_selected = int(steps['selector'].get_support(indices=False).sum())
                elif 'rfecv' in steps and hasattr(steps['rfecv'], 'support_'):
                    n_selected = int(steps['rfecv'].support_.sum())
            if n_selected is not None:
                print(f"[Fold {fold}] Integrated selector kept {n_selected} features")
                # Capture selected feature names
                if 'selector' in steps and hasattr(steps['selector'], 'get_support'):
                    mask = steps['selector'].get_support(indices=False)
                    selected_fold_features = list(np.array(X.columns)[mask])
                elif 'rfecv' in steps and hasattr(steps['rfecv'], 'support_'):
                    mask = steps['rfecv'].support_
                    selected_fold_features = list(np.array(X.columns)[mask])
                else:
                    selected_fold_features = []
                selected_features_per_fold.append(selected_fold_features)
                preview = ', '.join(selected_fold_features[:5])
                suffix = '...' if len(selected_fold_features) > 5 else ''
                print(f"[Fold {fold}] Selected features: {preview}{suffix}")
        else:
            # Pre-selected features path (All or Statistical)
            selected_features = clf_info['features']
            X_train_selected = X_train[selected_features]
            X_test_selected = X_test[selected_features]

            # Train and predict
            model_fitted = model.fit(X_train_selected, y_train)
            y_pred = model_fitted.predict(X_test_selected)
            if hasattr(model_fitted, 'predict_proba'):
                y_proba = model_fitted.predict_proba(X_test_selected)[:, 1]
            elif hasattr(model_fitted, 'decision_function'):
                y_proba = model_fitted.decision_function(X_test_selected)
            else:
                y_proba = (y_pred == 1).astype(float)

            # For feature importance, use model-based importance if available
            if hasattr(model_fitted, 'feature_importances_'):
                importances = pd.Series(model_fitted.feature_importances_, index=selected_features)
            elif hasattr(model_fitted, 'coef_') and hasattr(model_fitted.coef_, 'shape'):
                if len(model_fitted.coef_.shape) > 1:
                    importances = pd.Series(np.abs(model_fitted.coef_[0]), index=selected_features)
                else:
                    importances = pd.Series(np.abs(model_fitted.coef_), index=selected_features)
            elif hasattr(model_fitted, 'named_steps') and 'logisticregression' in getattr(model_fitted, 'named_steps', {}):
                coefs = model_fitted.named_steps['logisticregression'].coef_[0]
                importances = pd.Series(np.abs(coefs), index=selected_features)
            elif clf_name.endswith('(All)'):
                importances = X_train_selected.var()
            else:
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
        
        fold_metrics = calculate_metrics(y_test, y_pred, y_proba)
        for metric in metrics_history:
            metrics_history[metric].append(fold_metrics[metric])
        confusion_matrices.append(fold_metrics['Confusion_Matrix'])
    
    feature_importance_dict[clf_name] += model_feature_importance

    # After CV, analyze integrated selector stability and store a summary selection (optional)
    if clf_info.get('selector', None) == 'integrated' and selected_features_per_fold is not None:
        analyze_integrated_stability(selected_features_per_fold, clf_name)
        # Add features that appeared in at least one fold to features_summary_data
        fold_union = sorted(set([f for subset in selected_features_per_fold for f in subset]))
        for i, feat in enumerate(fold_union):
            features_summary_data.append({
                'Selection_Method': clf_name,
                'Feature_Name': feat,
                'Rank': i + 1,
                'Selection_Type': 'Integrated'
            })
    
    # Calculate average metrics and store results
    avg_metrics = {}
    std_metrics = {}
    for metric in metrics_history:
        avg_metrics[metric] = np.mean(metrics_history[metric])
        std_metrics[metric] = np.std(metrics_history[metric])
    
    # Store results for this classifier
    # Try to capture selected feature count if integrated
    selected_features_count = None
    if clf_info.get('selector', None) == 'integrated':
        try:
            if 'selector' in model_fitted.named_steps and hasattr(model_fitted.named_steps['selector'], 'get_support'):
                selected_features_count = int(model_fitted.named_steps['selector'].get_support(indices=False).sum())
            elif 'rfecv' in model_fitted.named_steps and hasattr(model_fitted.named_steps['rfecv'], 'support_'):
                selected_features_count = int(model_fitted.named_steps['rfecv'].support_.sum())
        except Exception:
            selected_features_count = None

    fold_metrics_copy = {m: list(vals) for m, vals in metrics_history.items()}

    all_classifier_results[clf_name] = {
        'avg_metrics': avg_metrics,
        'std_metrics': std_metrics,
        'top_features': (model_feature_importance / kf.cvargs['n_splits']).sort_values(ascending=False).head(10),
        'confusion_matrix': np.sum(confusion_matrices, axis=0),
        'total_cv_folds': len(metrics_history['Accuracy']),
        'selected_features_count': selected_features_count,
        'fold_metrics': fold_metrics_copy
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
    'True_Positive': results['confusion_matrix'][1,1],
    'Selected_Features_Count': results.get('selected_features_count', None)
    }
    metrics_data.append(row)

# Save to CSV
metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_csv(csv_filename, index=False)
print(f"Metrics summary saved to CSV: {csv_filename}")

print("=" * 80)

# Save a CSV with selected features summary (statistical + integrated)
try:
    features_df = pd.DataFrame(features_summary_data)
    features_csv = f"selected_features_summary_{timestamp}.csv"
    features_df.to_csv(features_csv, index=False)
    print(f"Selected features summary saved to: {features_csv}")
except Exception as e:
    print(f"Warning: could not save selected features summary CSV ({e})")



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