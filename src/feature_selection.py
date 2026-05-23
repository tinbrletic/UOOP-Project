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
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, RFECV
from skrebate import ReliefF
from imblearn.over_sampling import SMOTE
from scipy.stats import wilcoxon, mannwhitneyu, kruskal, chi2_contingency, ks_2samp, friedmanchisquare, norm, rankdata
import itertools
import re
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
from contextlib import contextmanager
import datetime
import os

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# SMOTE is now applied per-fold inside the CV loop (training data only),
# so we load the ORIGINAL unbalanced dataset here. Test folds keep the real
# class distribution; only training folds are oversampled with SMOTE.
dataset_file = 'data/processed/peptide_baza_formatted.csv'
# dataset_file = 'data/processed/peptide_baza_balanced.csv'  # legacy: pre-balanced (causes test-fold leakage)

# --- Caching imports and configuration ---
import os, json, pickle, hashlib, time, sys, platform
from joblib import Memory

CACHE_DIR = "cache"
JOBLIB_DIR = os.path.join(CACHE_DIR, "joblib")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(JOBLIB_DIR, exist_ok=True)

PARQUET_PATH = dataset_file.replace(".csv", ".parquet")
FINGERPRINT_JSON = os.path.join(CACHE_DIR, "dataset_fingerprint.json")

memory = Memory(location=JOBLIB_DIR, verbose=0)

def dataset_fingerprint(path):
    try:
        st = os.stat(path)
        return {"path": os.path.abspath(path), "size": st.st_size, "mtime": st.st_mtime}
    except FileNotFoundError:
        return {"path": os.path.abspath(path), "size": 0, "mtime": 0.0}

def model_specs(classifiers):
    specs = {}
    for name, info in classifiers.items():
        mdl = info["model"]
        params = getattr(mdl, "get_params", lambda: {})()
        specs[name] = str(sorted(params.items()))
    return specs

def make_run_id(dataset_file, classifiers, cv, extra=None):
    import sklearn
    payload = {
        "dataset": dataset_fingerprint(dataset_file),
        "classifiers": model_specs(classifiers),
        "cv": {"splits": cv.cvargs["n_splits"], "repeats": getattr(cv, "n_repeats", 1), "seed": cv.random_state},
        "extra": {
            "python": sys.version,
            "platform": platform.platform(),
            "sklearn": getattr(sklearn, "__version__", "unknown"),
            **(extra or {})
        },
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()

def cache_paths(run_id):
    return (os.path.join(CACHE_DIR, f"results_{run_id}.pkl"),
            os.path.join(CACHE_DIR, f"featimp_{run_id}.pkl"),
            os.path.join(CACHE_DIR, f"oof_{run_id}.npz"))

@memory.cache
def cached_stat_select(X_df, y_ser, method, top_k, alpha, adjust, es_min, corr_filter=False, corr_threshold=0.95):
    return statistical_feature_selection(X_df, y_ser, method=method, top_k=top_k, alpha=alpha, adjust=adjust, es_min=es_min)

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

@contextmanager
def catch_convergence():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        yield w


def make_rfecv_logreg_l1(
    C=0.1,
    class_weight='balanced',
    random_state=42,
    max_iter=50000,
    tol=1e-3,
    step=0.1,
    cv=5,
    n_jobs=-1,
    solver='saga',
    fallback_solver='liblinear'
):
    # Primary estimator
    base_est = LogisticRegression(
        penalty='l1',
        solver=solver,
        C=C,
        class_weight=class_weight,
        random_state=random_state,
        max_iter=max_iter,
        tol=tol,
        n_jobs=(n_jobs if solver == 'liblinear' else None)
    )

    class RFECVWithFallback(RFECV):
        def fit(self, X, y):
            # First attempt
            with catch_convergence() as wlist:
                result = super().fit(X, y)
            conv = any(issubclass(w.category, ConvergenceWarning) for w in wlist)
            if conv and fallback_solver and fallback_solver != solver:
                print(f"[RFECV] ConvergenceWarning detected with solver='{solver}'. Retrying with fallback solver='{fallback_solver}'.")
                fb_est = LogisticRegression(
                    penalty='l1',
                    solver=fallback_solver,
                    C=C,
                    class_weight=class_weight,
                    random_state=random_state,
                    max_iter=max_iter,
                    tol=tol,
                    n_jobs=(n_jobs if fallback_solver == 'liblinear' else None)
                )
                self.estimator = fb_est
                # Clean attributes from previous fit if present
                for attr in ['grid_scores_', 'cv_results_', 'ranking_', 'support_', 'n_features_']:
                    if hasattr(self, attr):
                        try:
                            delattr(self, attr)
                        except Exception:
                            pass
                # Second attempt with fallback
                with catch_convergence() as _:
                    result = super(RFECVWithFallback, self).fit(X, y)
            # Final solver info
            try:
                est = getattr(self, 'estimator_', None) or self.estimator
                print(f"[RFECV] Final solver used: {getattr(est, 'solver', 'unknown')}")
            except Exception:
                pass
            return result

    # Use n_jobs=1 to keep RFECV inner-CV fits in-process so ConvergenceWarning capture works reliably
    rfecv_fb = RFECVWithFallback(
        estimator=base_est, step=step, cv=cv, scoring='roc_auc', n_jobs=1, min_features_to_select=1
    )
    return rfecv_fb


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

        # Statistical selector (StatSelect) path: treat selected features with uniform weight
        if 'stat' in steps and hasattr(steps['stat'], 'selected_') and steps['stat'].selected_:
            sel = steps['stat'].selected_
            return pd.Series({c: 1.0 for c in sel})

        # Direct classifier importances (if no selector step)
        if 'clf' in steps:
            clf = steps['clf']
            if hasattr(clf, 'feature_importances_'):
                return pd.Series(clf.feature_importances_, index=X_columns)
            if hasattr(clf, 'coef_'):
                coefs = clf.coef_[0] if len(clf.coef_.shape) > 1 else clf.coef_.ravel()
                return pd.Series(np.abs(coefs), index=X_columns)

    return pd.Series(index=X_columns, data=0.0)


# Formatting helper for p-values
def fmt_p(p):
    try:
        p = float(p)
        return f"{p:.3e}" if p < 1e-4 else f"{p:.4f}"
    except Exception:
        return str(p)


class StatSelect(BaseEstimator, TransformerMixin):
    def __init__(self, method='mann_whitney', top_k=15, alpha=0.05, adjust='bh',
                 es_min=0.10, corr_filter=False, corr_threshold=0.95, random_state=42):
        self.method = method
        self.top_k = top_k
        self.alpha = alpha
        self.adjust = adjust
        self.es_min = es_min
        self.corr_filter = corr_filter
        self.corr_threshold = corr_threshold
        self.random_state = random_state
        self.selected_ = []
        self.columns_ = None

    def _corr_prune(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.corr_filter:
            return X
        corr = X.corr(method='spearman').abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > self.corr_threshold)]
        return X.drop(columns=to_drop) if len(to_drop) else X

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        Xw = self._corr_prune(X)
        # Align y index to X to avoid label-mismatch during .loc access in selection
        y_series = pd.Series(y)
        if len(y_series) == len(Xw):
            try:
                y_series.index = Xw.index
            except Exception:
                pass
        feats, _ = statistical_feature_selection(
            Xw, y_series, method=self.method, top_k=self.top_k,
            alpha=self.alpha, adjust=self.adjust, es_min=self.es_min
        )
        self.selected_ = feats
        print(f"[StatSelect] method={self.method}, adjust={self.adjust}, es_min={self.es_min:.2f}, selected={len(self.selected_)}")
        return self

    def transform(self, X):
        if not self.selected_:
            return X
        if isinstance(X, pd.DataFrame):
            cols = [c for c in self.selected_ if c in X.columns]
            return X[cols] if cols else X
        # numpy array fallback uses original column order
        if self.columns_:
            idx = [self.columns_.index(c) for c in self.selected_ if c in self.columns_]
            return X[:, idx] if idx else X
        return X

    # Optional compatibility with get_support
    def get_support(self, indices=False):
        if self.columns_ is None:
            return [] if indices else np.array([], dtype=bool)
        mask = np.array([c in set(self.selected_) for c in self.columns_], dtype=bool)
        return np.where(mask)[0] if indices else mask


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

        # Fast IO: prefer Parquet if fingerprint matches
        fp_now = dataset_fingerprint(dataset_file)
        fp_old = None
        if os.path.exists(FINGERPRINT_JSON):
            try:
                with open(FINGERPRINT_JSON, 'r') as fpf:
                    fp_db = json.load(fpf)
                    fp_old = fp_db.get(os.path.abspath(dataset_file))
            except Exception:
                fp_old = None

        use_parquet = os.path.exists(PARQUET_PATH) and (fp_old == fp_now)
        if use_parquet:
            try:
                data = pd.read_parquet(PARQUET_PATH)
                print(f"[CACHE] Loaded dataset from Parquet: {PARQUET_PATH}")
            except Exception:
                data = pd.read_csv(dataset_file, sep=';', quotechar='"')
        else:
            data = pd.read_csv(dataset_file, sep=';', quotechar='"')
            try:
                data.to_parquet(PARQUET_PATH, index=False)
                # update fingerprint registry
                fp_db = {}
                if os.path.exists(FINGERPRINT_JSON):
                    try:
                        with open(FINGERPRINT_JSON, 'r') as fpf:
                            fp_db = json.load(fpf)
                    except Exception:
                        fp_db = {}
                fp_db[os.path.abspath(dataset_file)] = fp_now
                with open(FINGERPRINT_JSON, 'w') as fpf:
                    json.dump(fp_db, fpf, indent=2)
                print(f"[CACHE] Wrote Parquet and updated fingerprint.")
            except Exception as e:
                print(f"[CACHE] Parquet write skipped: {e}")
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
    # Paired design: each fold scores every model, so the omnibus test is Friedman
    # (see run_friedman above) and the matching post-hoc is the Friedman-Conover variant.
    # Newer scikit-posthocs requires val_col/group_col for plain `posthoc_conover` on DataFrames;
    # `posthoc_conover_friedman` accepts wide-format directly (rows=blocks, cols=treatments).
    try:
        ph = sp.posthoc_conover_friedman(metric_df, p_adjust=p_adjust_method)
    except AttributeError:
        # Older scikit-posthocs (<0.6.0): function didn't exist; fall back to list-of-arrays form.
        ranks = metric_df.rank(axis=1, ascending=False, method='average')
        ph = sp.posthoc_conover([ranks[c].values for c in ranks.columns], p_adjust=p_adjust_method)
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

def statistical_feature_selection(X, y, method='mann_whitney', top_k=10, alpha=0.05, adjust='bh', es_min=0.10):
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
    
    # Apply p-adjust and effect-size filtering
    pvals = np.array([d.get('p_value', 1.0) for d in results.values()], dtype=float)
    if pvals.size:
        adj = p_adjust(pvals, method=adjust)
        for (feat, d), p_adj in zip(results.items(), adj):
            d['p_adj'] = float(p_adj)
    else:
        for d in results.values():
            d['p_adj'] = 1.0

    def _passes_effect(d, method_name):
        es = abs(d.get('effect_size', 0.0))
        if method_name in ('mann_whitney', 'wilcoxon', 'kruskal', 'ks_2samp'):
            return es >= es_min
        if method_name == 'chi2':
            return es >= es_min
        return True

    filtered = [(f, d.get('p_adj', 1.0), d.get('effect_size', 0.0))
                for f, d in results.items()
                if d.get('p_adj', 1.0) < alpha and _passes_effect(d, method)]
    filtered.sort(key=lambda x: (x[1], -abs(x[2])))
    selected_features = [f for f, _, _ in filtered[:top_k]]

    # Log results with p-adjust
    all_p = np.array([v['p_value'] for v in results.values() if 'p_value' in v], dtype=float)
    all_padj = np.array([v.get('p_adj', 1.0) for v in results.values()], dtype=float)
    if all_p.size:
        print(f"  Significant features found (raw): {(all_p < alpha).sum()}/{len(all_p)}")
        print(f"  Significant features found (adj): {(all_padj < alpha).sum()}/{len(all_padj)}")
        print(f"  Min raw p: {fmt_p(np.min(all_p))}, median p: {fmt_p(np.median(all_p))}")
        print(f"  Min p_adj: {fmt_p(np.min(all_padj))}, median p_adj: {fmt_p(np.median(all_padj))}")
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
mann_whitney_features, mw_results = cached_stat_select(X, y, 'mann_whitney', 15, 0.05, 'bh', 0.10, False, 0.95)
wilcoxon_features, w_results = cached_stat_select(X, y, 'wilcoxon', 15, 0.05, 'bh', 0.10, False, 0.95)
kruskal_features,  k_results = cached_stat_select(X, y, 'kruskal', 15, 0.05, 'bh', 0.10, False, 0.95)
chi2_features,     chi_results = cached_stat_select(X, y, 'chi2', 15, 0.05, 'bh', 0.10, False, 0.95)
ks_features,       ks_results  = cached_stat_select(X, y, 'ks_2samp', 15, 0.05, 'bh', 0.10, False, 0.95)

print(f"\nStatistical Feature Selection Results:")
print(f"Mann-Whitney selected: {len(mann_whitney_features)} features")
print(f"Wilcoxon selected: {len(wilcoxon_features)} features")
print(f"Kruskal-Wallis selected: {len(kruskal_features)} features")
print(f"Chi-square selected: {len(chi2_features)} features")
print(f"Kolmogorov-Smirnov selected: {len(ks_features)} features")

# Detailed lists of selected features for each statistical method
print(f"\nMANN_WHITNEY SELECTED FEATURES ({len(mann_whitney_features)}):")
for i, feature in enumerate(mann_whitney_features, 1):
    p_val = mw_results[feature].get('p_adj', mw_results[feature].get('p_value', 1.0))
    es = mw_results[feature].get('effect_size', 0.0)
    print(f"  {i:2d}. {feature:30} (p_adj={fmt_p(p_val)}, es={abs(es):.3f})")

print(f"\nWILCOXON SELECTED FEATURES ({len(wilcoxon_features)}):")
for i, feature in enumerate(wilcoxon_features, 1):
    p_val = w_results[feature].get('p_adj', w_results[feature].get('p_value', 1.0))
    es = w_results[feature].get('effect_size', 0.0)
    print(f"  {i:2d}. {feature:30} (p_adj={fmt_p(p_val)}, es={abs(es):.3f})")

print(f"\nKRUSKAL SELECTED FEATURES ({len(kruskal_features)}):")
for i, feature in enumerate(kruskal_features, 1):
    p_val = k_results[feature].get('p_adj', k_results[feature].get('p_value', 1.0))
    es = k_results[feature].get('effect_size', 0.0)
    print(f"  {i:2d}. {feature:30} (p_adj={fmt_p(p_val)}, es={abs(es):.3f})")

print(f"\nCHI2 SELECTED FEATURES ({len(chi2_features)}):")
for i, feature in enumerate(chi2_features, 1):
    p_val = chi_results[feature].get('p_adj', chi_results[feature].get('p_value', 1.0))
    es = chi_results[feature].get('effect_size', 0.0)
    print(f"  {i:2d}. {feature:30} (p_adj={fmt_p(p_val)}, es={abs(es):.3f})")

print(f"\nKS SELECTED FEATURES ({len(ks_features)}):")
for i, feature in enumerate(ks_features, 1):
    p_val = ks_results[feature].get('p_adj', ks_results[feature].get('p_value', 1.0))
    es = ks_results[feature].get('effect_size', 0.0)
    print(f"  {i:2d}. {feature:30} (p_adj={fmt_p(p_val)}, es={abs(es):.3f})")

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
    # Random Forest statistical subsets (in-CV selection to avoid leakage)
    "Random Forest (MW - inCV)": {
        'model': Pipeline([
            ('stat', StatSelect(method='mann_whitney', top_k=15, alpha=0.05, adjust='bh', es_min=0.10, corr_filter=False)),
            ('clf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1))
        ]),
        'selector': 'integrated',
        'features': None
    },
    "Random Forest (Wilcoxon - inCV)": {
        'model': Pipeline([
            ('stat', StatSelect(method='wilcoxon', top_k=15, alpha=0.05, adjust='bh', es_min=0.10, corr_filter=False)),
            ('clf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1))
        ]),
        'selector': 'integrated',
        'features': None
    },
    "Random Forest (Kruskal - inCV)": {
        'model': Pipeline([
            ('stat', StatSelect(method='kruskal', top_k=15, alpha=0.05, adjust='bh', es_min=0.10, corr_filter=False)),
            ('clf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1))
        ]),
        'selector': 'integrated',
        'features': None
    },
    "Random Forest (Chi-square - inCV)": {
        'model': Pipeline([
            ('stat', StatSelect(method='chi2', top_k=15, alpha=0.05, adjust='bh', es_min=0.10, corr_filter=False)),
            ('clf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1))
        ]),
        'selector': 'integrated',
        'features': None
    },
    "Random Forest (KS - inCV)": {
        'model': Pipeline([
            ('stat', StatSelect(method='ks_2samp', top_k=15, alpha=0.05, adjust='bh', es_min=0.10, corr_filter=False)),
            ('clf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1))
        ]),
        'selector': 'integrated',
        'features': None
    },
    # Logistic Regression statistical subsets
    "Logistic Regression (MW - inCV)": {
        'model': Pipeline([
            ('stat', StatSelect(method='mann_whitney', top_k=15, alpha=0.05, adjust='bh', es_min=0.10, corr_filter=False)),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=5000, class_weight='balanced', solver='saga', penalty='l1', C=0.1, random_state=42, n_jobs=-1))
        ]),
        'selector': 'integrated',
        'features': None
    },
    "Logistic Regression (Wilcoxon - inCV)": {
        'model': Pipeline([
            ('stat', StatSelect(method='wilcoxon', top_k=15, alpha=0.05, adjust='bh', es_min=0.10, corr_filter=False)),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=5000, class_weight='balanced', solver='saga', penalty='l1', C=0.1, random_state=42, n_jobs=-1))
        ]),
        'selector': 'integrated',
        'features': None
    },
    "Logistic Regression (Kruskal - inCV)": {
        'model': Pipeline([
            ('stat', StatSelect(method='kruskal', top_k=15, alpha=0.05, adjust='bh', es_min=0.10, corr_filter=False)),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=5000, class_weight='balanced', solver='saga', penalty='l1', C=0.1, random_state=42, n_jobs=-1))
        ]),
        'selector': 'integrated',
        'features': None
    },
    "Logistic Regression (Chi-square - inCV)": {
        'model': Pipeline([
            ('stat', StatSelect(method='chi2', top_k=15, alpha=0.05, adjust='bh', es_min=0.10, corr_filter=False)),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=5000, class_weight='balanced', solver='saga', penalty='l1', C=0.1, random_state=42, n_jobs=-1))
        ]),
        'selector': 'integrated',
        'features': None
    },
    "Logistic Regression (KS - inCV)": {
        'model': Pipeline([
            ('stat', StatSelect(method='ks_2samp', top_k=15, alpha=0.05, adjust='bh', es_min=0.10, corr_filter=False)),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=5000, class_weight='balanced', solver='saga', penalty='l1', C=0.1, random_state=42, n_jobs=-1))
        ]),
        'selector': 'integrated',
        'features': None
    },
    # SVM statistical subsets
    "SVM (MW - inCV)": {
        'model': Pipeline([
            ('stat', StatSelect(method='mann_whitney', top_k=15, alpha=0.05, adjust='bh', es_min=0.10, corr_filter=False)),
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel='rbf', C=0.5, probability=True, random_state=42))
        ]),
        'selector': 'integrated',
        'features': None
    },
    "SVM (Wilcoxon - inCV)": {
        'model': Pipeline([
            ('stat', StatSelect(method='wilcoxon', top_k=15, alpha=0.05, adjust='bh', es_min=0.10, corr_filter=False)),
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel='rbf', C=0.5, probability=True, random_state=42))
        ]),
        'selector': 'integrated',
        'features': None
    },
    "SVM (Kruskal - inCV)": {
        'model': Pipeline([
            ('stat', StatSelect(method='kruskal', top_k=15, alpha=0.05, adjust='bh', es_min=0.10, corr_filter=False)),
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel='rbf', C=0.5, probability=True, random_state=42))
        ]),
        'selector': 'integrated',
        'features': None
    },
    "SVM (Chi-square - inCV)": {
        'model': Pipeline([
            ('stat', StatSelect(method='chi2', top_k=15, alpha=0.05, adjust='bh', es_min=0.10, corr_filter=False)),
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel='rbf', C=0.5, probability=True, random_state=42))
        ]),
        'selector': 'integrated',
        'features': None
    },
    "SVM (KS - inCV)": {
        'model': Pipeline([
            ('stat', StatSelect(method='ks_2samp', top_k=15, alpha=0.05, adjust='bh', es_min=0.10, corr_filter=False)),
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel='rbf', C=0.5, probability=True, random_state=42))
        ]),
        'selector': 'integrated',
        'features': None
    },
    # K-Neighbors statistical subsets
    "K-Neighbors (MW - inCV)": {
        'model': Pipeline([
            ('stat', StatSelect(method='mann_whitney', top_k=15, alpha=0.05, adjust='bh', es_min=0.10, corr_filter=False)),
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=15, algorithm='kd_tree'))
        ]),
        'selector': 'integrated',
        'features': None
    },
    "K-Neighbors (Wilcoxon - inCV)": {
        'model': Pipeline([
            ('stat', StatSelect(method='wilcoxon', top_k=15, alpha=0.05, adjust='bh', es_min=0.10, corr_filter=False)),
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=15, algorithm='kd_tree'))
        ]),
        'selector': 'integrated',
        'features': None
    },
    "K-Neighbors (Kruskal - inCV)": {
        'model': Pipeline([
            ('stat', StatSelect(method='kruskal', top_k=15, alpha=0.05, adjust='bh', es_min=0.10, corr_filter=False)),
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=15, algorithm='kd_tree'))
        ]),
        'selector': 'integrated',
        'features': None
    },
    "K-Neighbors (Chi-square - inCV)": {
        'model': Pipeline([
            ('stat', StatSelect(method='chi2', top_k=15, alpha=0.05, adjust='bh', es_min=0.10, corr_filter=False)),
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=15, algorithm='kd_tree'))
        ]),
        'selector': 'integrated',
        'features': None
    },
    "K-Neighbors (KS - inCV)": {
        'model': Pipeline([
            ('stat', StatSelect(method='ks_2samp', top_k=15, alpha=0.05, adjust='bh', es_min=0.10, corr_filter=False)),
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=15, algorithm='kd_tree'))
        ]),
        'selector': 'integrated',
        'features': None
    },
    # Decision Tree statistical subsets
    "Decision Tree (MW - inCV)": {
        'model': Pipeline([
            ('stat', StatSelect(method='mann_whitney', top_k=15, alpha=0.05, adjust='bh', es_min=0.10, corr_filter=False)),
            ('clf', DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42))
        ]),
        'selector': 'integrated',
        'features': None
    },
    "Decision Tree (Wilcoxon - inCV)": {
        'model': Pipeline([
            ('stat', StatSelect(method='wilcoxon', top_k=15, alpha=0.05, adjust='bh', es_min=0.10, corr_filter=False)),
            ('clf', DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42))
        ]),
        'selector': 'integrated',
        'features': None
    },
    "Decision Tree (Kruskal - inCV)": {
        'model': Pipeline([
            ('stat', StatSelect(method='kruskal', top_k=15, alpha=0.05, adjust='bh', es_min=0.10, corr_filter=False)),
            ('clf', DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42))
        ]),
        'selector': 'integrated',
        'features': None
    },
    "Decision Tree (Chi-square - inCV)": {
        'model': Pipeline([
            ('stat', StatSelect(method='chi2', top_k=15, alpha=0.05, adjust='bh', es_min=0.10, corr_filter=False)),
            ('clf', DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42))
        ]),
        'selector': 'integrated',
        'features': None
    },
    "Decision Tree (KS - inCV)": {
        'model': Pipeline([
            ('stat', StatSelect(method='ks_2samp', top_k=15, alpha=0.05, adjust='bh', es_min=0.10, corr_filter=False)),
            ('clf', DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42))
        ]),
        'selector': 'integrated',
        'features': None
    },
    # Naive Bayes statistical subsets
    "Naive Bayes (MW - inCV)": {
        'model': make_pipeline(StatSelect(method='mann_whitney', top_k=15, alpha=0.05, adjust='bh', es_min=0.10, corr_filter=False), StandardScaler(), GaussianNB()),
        'selector': 'integrated',
        'features': None
    },
    "Naive Bayes (Wilcoxon - inCV)": {
        'model': make_pipeline(StatSelect(method='wilcoxon', top_k=15, alpha=0.05, adjust='bh', es_min=0.10, corr_filter=False), StandardScaler(), GaussianNB()),
        'selector': 'integrated',
        'features': None
    },
    "Naive Bayes (Kruskal - inCV)": {
        'model': make_pipeline(StatSelect(method='kruskal', top_k=15, alpha=0.05, adjust='bh', es_min=0.10, corr_filter=False), StandardScaler(), GaussianNB()),
        'selector': 'integrated',
        'features': None
    },
    "Naive Bayes (Chi-square - inCV)": {
        'model': make_pipeline(StatSelect(method='chi2', top_k=15, alpha=0.05, adjust='bh', es_min=0.10, corr_filter=False), StandardScaler(), GaussianNB()),
        'selector': 'integrated',
        'features': None
    },
    "Naive Bayes (KS - inCV)": {
        'model': make_pipeline(StatSelect(method='ks_2samp', top_k=15, alpha=0.05, adjust='bh', es_min=0.10, corr_filter=False), StandardScaler(), GaussianNB()),
        'selector': 'integrated',
        'features': None
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
            ('rfecv', make_rfecv_logreg_l1(
                C=0.1, cv=5, max_iter=50000, tol=1e-3, solver='saga', fallback_solver='liblinear', n_jobs=-1
            )),
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
            ('rfecv', make_rfecv_logreg_l1(
                C=0.1, cv=5, max_iter=50000, tol=1e-3, solver='saga', fallback_solver='liblinear', n_jobs=-1
            )),
            ('clf', KNeighborsClassifier(n_neighbors=15, algorithm='kd_tree'))
        ]),
        'selector': 'integrated',
        'features': None
    },
    # RFECV (L1-LR) -> Naive Bayes
    "Integrated-RFECV (L1-LR) + Naive Bayes": {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('rfecv', make_rfecv_logreg_l1(
                C=0.1, cv=5, max_iter=50000, tol=1e-3, solver='saga', fallback_solver='liblinear', n_jobs=-1
            )),
            ('clf', GaussianNB())
        ]),
        'selector': 'integrated',
        'features': None
    }
}

# Configure cross-validation
kf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)

# Optional deterministic splits cache
splits_file = os.path.join(CACHE_DIR, f"splits_{kf.cvargs['n_splits']}x{kf.n_repeats}_seed{kf.random_state}.npz")

def _splits_valid(splits_list, n_samples):
    try:
        for tr, te in splits_list:
            if len(tr) == 0 or len(te) == 0:
                return False
            if int(np.max(tr)) >= n_samples or int(np.max(te)) >= n_samples:
                return False
            if int(np.min(tr)) < 0 or int(np.min(te)) < 0:
                return False
        return True
    except Exception:
        return False

if os.path.exists(splits_file):
    try:
        data_splits = np.load(splits_file, allow_pickle=True)
        splits = list(data_splits["splits"])  # list of (train_idx, test_idx)
        if not _splits_valid(splits, len(X)):
            print(f"[CACHE] Cached splits invalid for current dataset (size changed). Regenerating.")
            splits = list(kf.split(X, y))
            np.savez_compressed(splits_file, splits=np.array(splits, dtype=object))
        else:
            print(f"[CACHE] Loaded CV splits from {splits_file}")
    except Exception:
        splits = list(kf.split(X, y))
        try:
            np.savez_compressed(splits_file, splits=np.array(splits, dtype=object))
        except Exception:
            pass
else:
    splits = list(kf.split(X, y))
    try:
        np.savez_compressed(splits_file, splits=np.array(splits, dtype=object))
        print(f"[CACHE] Saved CV splits to {splits_file}")
    except Exception as e:
        print(f"[CACHE] Could not save splits: {e}")

# Log cross-validation configuration
log_cv_configuration(kf)

# Log classifier configurations
log_classifier_configuration(classifiers)

# Global feature importance dictionary (baseline + statistical subsets)
feature_importance_dict = {name: pd.Series(0.0, index=X.columns) for name in classifiers.keys()}

# Initialize results dictionary to store all classifier performance
all_classifier_results = {}

# Run-level cache gate
no_cache = os.environ.get("NO_CACHE", "0") == "1" or "--no-cache" in sys.argv
run_id = make_run_id(dataset_file, classifiers, kf, extra={
    "stat_select": {"top_k": 15, "alpha": 0.05, "adjust": "bh", "es_min": 0.10},
    "smote_in_cv": {"k_neighbors": 5, "random_state": 42, "sampling_strategy": "auto"},
})
res_pkl, feat_pkl, oof_npz = cache_paths(run_id)

feature_summary_pkl = os.path.join(CACHE_DIR, f"features_summary_{run_id}.pkl")

if (not no_cache) and os.path.exists(res_pkl) and os.path.exists(feat_pkl):
    with open(res_pkl, "rb") as f:
        all_classifier_results = pickle.load(f)
    with open(feat_pkl, "rb") as f:
        feature_importance_dict = pickle.load(f)
    try:
        with open(feature_summary_pkl, "rb") as f:
            features_summary_data = pickle.load(f)
    except Exception:
        pass
    print(f"[CACHE] Učitao rezultate iz cachea, preskačem treniranje. run_id={run_id}")
else:
    # SMOTE inside CV: oversample ONLY the training fold so the model sees a balanced
    # class distribution during fit, while the test fold keeps the original unbalanced
    # distribution (= real-world performance estimate, no leakage of synthetic samples).
    # Pre-compute once per fold (SMOTE is deterministic at random_state=42) and reuse
    # across all classifiers, since splits are shared.
    print("\n" + "=" * 60)
    print("SMOTE-in-CV: pre-computing balanced training folds")
    print("(test folds keep the original unbalanced distribution)")
    print("=" * 60)
    smote_folds = {}
    for _fold_idx, (_train_idx_pre, _) in enumerate(splits, 1):
        X_tr_pre = X.iloc[_train_idx_pre]
        y_tr_pre = y.iloc[_train_idx_pre]
        n_pos_before = int((y_tr_pre == 1).sum())
        n_neg_before = int((y_tr_pre == 0).sum())
        try:
            _smote = SMOTE(random_state=42, k_neighbors=5)
            X_tr_bal, y_tr_bal = _smote.fit_resample(X_tr_pre, y_tr_pre)
        except ValueError as _smote_err:
            print(f"[Fold {_fold_idx}] SMOTE failed ({_smote_err}); using original training data.")
            X_tr_bal = X_tr_pre.copy()
            y_tr_bal = y_tr_pre.copy()
        # imblearn preserves DataFrame/Series when inputs are DataFrame/Series; coerce defensively
        if not isinstance(X_tr_bal, pd.DataFrame):
            X_tr_bal = pd.DataFrame(X_tr_bal, columns=X.columns)
        if not isinstance(y_tr_bal, pd.Series):
            y_tr_bal = pd.Series(y_tr_bal, name=(y.name if y.name else 'targetcol'))
        # Reset indices so StatSelect's .loc-based class indexing inside in-CV pipelines works
        X_tr_bal = X_tr_bal.reset_index(drop=True)
        y_tr_bal = y_tr_bal.reset_index(drop=True)
        smote_folds[_fold_idx] = (X_tr_bal, y_tr_bal)
        # Log first 3 folds + last as a sanity check; suppress the rest to keep output readable
        if _fold_idx <= 3 or _fold_idx == len(splits):
            n_pos_after = int((y_tr_bal == 1).sum())
            n_neg_after = int((y_tr_bal == 0).sum())
            print(f"[Fold {_fold_idx}] train  before: pos={n_pos_before:4d} neg={n_neg_before:4d}"
                  f"   ->   after SMOTE: pos={n_pos_after:4d} neg={n_neg_after:4d}")
    print(f"SMOTE pre-computation complete for {len(smote_folds)} folds.\n")

    for clf_name, clf_info in classifiers.items():
        print(f"\n{'='*40}\nEvaluating {clf_name}\n{'='*40}")

        metrics_history = {metric: [] for metric in ['Accuracy', 'F1', 'AUC-ROC', 'Precision', 'Recall', 'MCC']}
        # OOF predictions vector for this model (sized to ORIGINAL dataset; test_idx positions only)
        y_oof = np.full(len(y), np.nan, dtype=float)
        # Accumulate per-fold feature importance for this classifier.
        model_feature_importance = pd.Series(0.0, index=X.columns)
        confusion_matrices = []
        # Track selected features per fold for integrated selectors
        selected_features_per_fold = [] if clf_info.get('selector', None) == 'integrated' else None

        for fold, (train_idx, test_idx) in enumerate(splits, 1):
            # Training data: SMOTE-balanced (pre-computed once per fold above)
            X_train, y_train = smote_folds[fold]
            # Test data: ORIGINAL unbalanced fold — preserves real class distribution
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]

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

                # Save out-of-fold predictions for current test index
                y_oof[test_idx] = y_proba

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

                # Save out-of-fold predictions for current test index
                y_oof[test_idx] = y_proba

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

        # OOF ROC and AUC for this model
        mask = ~np.isnan(y_oof)
        if mask.sum() > 0 and np.unique(y.values[mask]).size == 2:
            fpr, tpr, thresholds = roc_curve(y.values[mask], y_oof[mask])
            roc_auc_val = auc(fpr, tpr)
        else:
            fpr, tpr, roc_auc_val = np.array([0.0, 1.0]), np.array([0.0, 1.0]), float('nan')

        all_classifier_results[clf_name] = {
            'avg_metrics': avg_metrics,
            'std_metrics': std_metrics,
            'top_features': (model_feature_importance / kf.cvargs['n_splits']).sort_values(ascending=False).head(10),
            'confusion_matrix': np.sum(confusion_matrices, axis=0),
            'total_cv_folds': len(metrics_history['Accuracy']),
            'selected_features_count': selected_features_count,
            'fold_metrics': fold_metrics_copy,
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': float(roc_auc_val)
            }
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

    # Save run-level results to cache
    try:
        with open(res_pkl, "wb") as f:
            pickle.dump(all_classifier_results, f)
        with open(feat_pkl, "wb") as f:
            pickle.dump(feature_importance_dict, f)
        with open(feature_summary_pkl, "wb") as f:
            pickle.dump(features_summary_data, f)
        print(f"[CACHE] Spremio rezultate. run_id={run_id}")
    except Exception as e:
        print(f"[CACHE] Save failed: {e}")

print("\n" + "="*80)
print("EXPERIMENT RESULTS SUMMARY")
print("="*80)
print(f"Experiment completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Dataset: {dataset_file}")
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

# --- OOF ROC plotting ---
def slugify(s):
    return re.sub(r'[^A-Za-z0-9]+', '_', s).strip('_')[:60]

ts_plot = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# All outputs (PNGs, CSVs, TXT, stats files) for this run land here.
# Folder name encodes the experimental variant so legacy "pre-balanced dataset" runs
# stay separated from the new SMOTE-inside-CV runs.
OUTPUT_DIR = os.path.join("results", "smote_in_cv", ts_plot)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"[OUTPUT] Per-run artifacts will be written to: {OUTPUT_DIR}")

# Combined ROC for all models
try:
    import importlib, os
    # Force a non-GUI backend to avoid Tk/Qt dependencies on headless/Windows
    os.environ.setdefault('MPLBACKEND', 'Agg')
    mpl = importlib.import_module('matplotlib')
    try:
        mpl.use('Agg')
    except Exception:
        pass
    plt = importlib.import_module('matplotlib.pyplot')
    plt.figure(figsize=(7, 6))
    for name, res in all_classifier_results.items():
        roc_info = res.get('roc_curve')
        if not roc_info:
            continue
        fpr = np.array(roc_info['fpr'], dtype=float)
        tpr = np.array(roc_info['tpr'], dtype=float)
        auc_val = float(roc_info['auc']) if 'auc' in roc_info else float('nan')
        if fpr.size > 1 and tpr.size > 1 and np.all(np.isfinite(fpr)) and np.all(np.isfinite(tpr)):
            label = f"{name} (AUC={auc_val:.3f})" if np.isfinite(auc_val) else f"{name} (AUC=NaN)"
            plt.plot(fpr, tpr, lw=1.5, label=label)

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves (OOF predictions)')
    plt.legend(loc='lower right', fontsize=7)
    plt.tight_layout()
    _combined_roc_path = os.path.join(OUTPUT_DIR, f"roc_curves_all_models_{ts_plot}.png")
    plt.savefig(_combined_roc_path, dpi=200)
    plt.close()
    print(f"Saved: {_combined_roc_path}")

    # Optional: per-model ROC plots
    for name, res in all_classifier_results.items():
        roc_info = res.get('roc_curve')
        if not roc_info:
            continue
        fpr = np.array(roc_info['fpr'], dtype=float)
        tpr = np.array(roc_info['tpr'], dtype=float)
        auc_val = float(roc_info['auc']) if 'auc' in roc_info else float('nan')
        if fpr.size <= 1 or tpr.size <= 1 or not np.all(np.isfinite(fpr)) or not np.all(np.isfinite(tpr)):
            continue
        plt.figure(figsize=(6, 5))
        label = f"AUC={auc_val:.3f}" if np.isfinite(auc_val) else "AUC=NaN"
        plt.plot(fpr, tpr, lw=1.8, label=label)
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC: {name}')
        plt.legend(loc='lower right')
        plt.tight_layout()
        fname = os.path.join(OUTPUT_DIR, f"roc_{slugify(name)}_{ts_plot}.png")
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"Saved: {fname}")

    # --- Grouped ROC plots by model families ---
    def plot_group_rocs(all_results, groups, ts_plot):
        import numpy as np
        import matplotlib.pyplot as plt

        def _match_any(name, substrings):
            # Require all provided substrings to be present to avoid overmatching across groups
            name_low = name.lower()
            return all(s.lower() in name_low for s in substrings)

        for group_name, substrings in groups.items():
            selected = []
            for name, res in all_results.items():
                if _match_any(name, substrings):
                    roc_info = res.get('roc_curve', {})
                    fpr = np.array(roc_info.get('fpr', []), dtype=float)
                    tpr = np.array(roc_info.get('tpr', []), dtype=float)
                    auc_val = float(roc_info.get('auc', float('nan'))) if 'auc' in roc_info else float('nan')
                    if fpr.size > 1 and tpr.size > 1 and np.all(np.isfinite(fpr)) and np.all(np.isfinite(tpr)):
                        selected.append((name, fpr, tpr, auc_val))

            if len(selected) == 0:
                print(f"[GROUP-ROC] Skipping '{group_name}' (no valid ROC curves).")
                continue

            plt.figure(figsize=(7, 6))
            for name, fpr, tpr, auc_val in selected:
                label = f"{name} (AUC={auc_val:.3f})" if np.isfinite(auc_val) else f"{name} (AUC=NaN)"
                plt.plot(fpr, tpr, lw=1.5, label=label)

            plt.plot([0, 1], [0, 1], 'k--', lw=1)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f"ROC (OOF) – {group_name}")
            plt.legend(loc='lower right', fontsize=7)
            plt.tight_layout()

            from re import sub
            def _slug(s):
                return sub(r'[^A-Za-z0-9]+', '_', s).strip('_')[:60]

            fname = os.path.join(OUTPUT_DIR, f"roc_group_{_slug(group_name)}_{ts_plot}.png")
            plt.savefig(fname, dpi=200)
            plt.close()
            print(f"[GROUP-ROC] Saved: {fname}")

    groups = {
        # Baseline full-feature models
        "Baseline (All features)": ["(All)"],

        # Statistical-selection in-CV variants by base learner
        "StatSelect – Random Forest": ["Random Forest (", "inCV)"],
        "StatSelect – Logistic Regression": ["Logistic Regression (", "inCV)"],
        "StatSelect – SVM": ["SVM (", "inCV)"],
        "StatSelect – K-Neighbors": ["K-Neighbors (", "inCV)"],
        "StatSelect – Decision Tree": ["Decision Tree (", "inCV)"],
        "StatSelect – Naive Bayes": ["Naive Bayes (", "inCV)"],

        # Integrated selectors
        "Integrated – L1-LR selector": ["Integrated-SelectFromModel (L1-LR selector)"],
        "Integrated – RF selector": ["Integrated-SelectFromModel (RF selector)"],
        "Integrated – RFECV (L1-LR)": ["Integrated-RFECV (L1-LR)"],
    }

    plot_group_rocs(all_classifier_results, groups, ts_plot)
except Exception as e:
    print(f"[WARN] Skipping ROC plotting due to: {e}")

# Save comprehensive results with timestamp
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# === Global statistical comparisons across models ===
for metric_name in ['AUC-ROC', 'F1']:
    out_prefix = os.path.join(OUTPUT_DIR, f"stats_{metric_name.replace('-', '_')}_{timestamp}")
    run_statistical_evaluation(
        all_classifier_results,
        metric_name=metric_name,
        alpha=0.05,
        p_adjust_method='holm',
        out_prefix=out_prefix
    )

results_filename = os.path.join(OUTPUT_DIR, f"feature_selection_results_{dataset_file}_smote_in_cv_{timestamp}.txt")

with open(results_filename, 'w') as f:
    f.write("PEPTIDE SYNTHESIS PREDICTION - FEATURE SELECTION RESULTS\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Experiment Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Dataset: {dataset_file}\n")
    f.write(f"Total samples (original): {len(y)}\n")
    f.write(f"Total features: {X.shape[1]}\n")
    f.write(f"Cross-validation: {kf.cvargs['n_splits']}-fold, {kf.n_repeats} repeats\n")
    f.write(f"Random state: {kf.random_state}\n")
    f.write(f"Balancing: SMOTE per-fold on TRAINING data only "
            f"(k_neighbors=5, random_state=42); test folds keep original unbalanced distribution.\n\n")
    
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
csv_filename = os.path.join(OUTPUT_DIR, f"feature_selection_metrics_{timestamp}.csv")
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
    features_csv = os.path.join(OUTPUT_DIR, f"selected_features_summary_{timestamp}.csv")
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