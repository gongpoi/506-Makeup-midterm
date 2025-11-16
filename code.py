import numpy as np
import pandas as pd
import pickle
import re
from collections import Counter
from datetime import datetime
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    HashingVectorizer,
    TfidfTransformer,
)
from scipy import sparse
import warnings
import sys
import json
from typing import Dict, Any, List
import os
import hashlib

warnings.filterwarnings("ignore")

# Configuration
USE_MANUAL_CLASS_WEIGHT = False
USE_ENTITY_AGG_FEATURES = False
N_SPLITS = 5
RANDOM_STATE = 42
MODEL_MAX_ITER = 2000

# Text vectorization configuration
USE_HASHING_VECTORIZER = False
TFIDF_MAX_FEATURES = 200000
TFIDF_MIN_DF = 3
TFIDF_MAX_DF = 0.98
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_SUBLINEAR_TF = True
HASHING_N_FEATURES = 2**18
HASHING_ALTERNATE_SIGN = False
TEXT_TRUNCATE = 8000

# C search range
C_GRID = list(np.logspace(-2, 0.5, 10))  # [0.01 .. ~3.16]
np.random.seed(RANDOM_STATE)

# Caching and consistency configuration
CACHE_DIR = "./cache"  # Unified cache directory for easier management
os.makedirs(CACHE_DIR, exist_ok=True)

FEATURE_COLUMNS_PKL = os.path.join(CACHE_DIR, "feature_columns.pkl")
SCALER_PKL = os.path.join(CACHE_DIR, "scaler.pkl")
AGG_STATS_PKL = os.path.join(CACHE_DIR, "aggregate_stats.pkl")
TEXT_VECTORIZER_PKL = os.path.join(CACHE_DIR, "text_vectorizer.pkl")
CV_RESULTS_PKL = os.path.join(CACHE_DIR, "cv_results_group.pkl")
FINAL_MODEL_PKL = os.path.join(CACHE_DIR, "logistic_regression_model.pkl")

# A simple configuration signature to detect changes in key settings
def config_signature() -> str:
    cfg = {
        "USE_HASHING_VECTORIZER": USE_HASHING_VECTORIZER,
        "TFIDF_MAX_FEATURES": TFIDF_MAX_FEATURES,
        "TFIDF_MIN_DF": TFIDF_MIN_DF,
        "TFIDF_MAX_DF": TFIDF_MAX_DF,
        "TFIDF_NGRAM_RANGE": TFIDF_NGRAM_RANGE,
        "TFIDF_SUBLINEAR_TF": TFIDF_SUBLINEAR_TF,
        "HASHING_N_FEATURES": HASHING_N_FEATURES,
        "HASHING_ALTERNATE_SIGN": HASHING_ALTERNATE_SIGN,
        "TEXT_TRUNCATE": TEXT_TRUNCATE,
        "USE_ENTITY_AGG_FEATURES": USE_ENTITY_AGG_FEATURES,
        "RANDOM_STATE": RANDOM_STATE,
        "N_SPLITS": N_SPLITS,
        "MODEL_MAX_ITER": MODEL_MAX_ITER,
        "C_GRID": tuple([float(x) for x in C_GRID]),
        "USE_MANUAL_CLASS_WEIGHT": USE_MANUAL_CLASS_WEIGHT,
    }
    raw = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(raw).hexdigest()

CURRENT_SIG = config_signature()

# Reading data and basic utilities
def safe_read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def clean_score_column(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    s = s.clip(lower=1, upper=5)
    return s.round().astype("Int64").astype("int32")

_whitespace_re = re.compile(r"\s+")
_html_tag_re = re.compile(r"<[^>]+>")

def clean_text(x: Any, max_len: int = 5000) -> str:
    if not isinstance(x, str):
        return ""
    x = _html_tag_re.sub(" ", x)
    x = _whitespace_re.sub(" ", x).strip()
    if len(x) > max_len:
        x = x[:max_len]
    return x

def compile_vocab_regex(vocab: List[str]):
    return [re.compile(rf"\b{re.escape(w)}\b", flags=re.IGNORECASE) for w in vocab]

positive_words = ["great","love","excellent","amazing","fantastic","wonderful","best","good","awesome"]
negative_words = ["bad","terrible","awful","horrible","worst","hate","disappointed","poor"]
POS_PATTERNS = compile_vocab_regex(positive_words)
NEG_PATTERNS = compile_vocab_regex(negative_words)

def count_vocab_words(text: str, patterns: List[re.Pattern]) -> int:
    if not text:
        return 0
    total = 0
    for p in patterns:
        total += len(p.findall(text))
    return int(total)

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize columns and fill missing values
    df = df.copy()
    must_have = ["Id","ProductId","UserId","Summary","Text","Score"]
    for c in must_have:
        if c not in df.columns:
            df[c] = np.nan
    df["Id"] = df["Id"].astype(str)
    for c in ["ProductId","UserId","Summary","Text"]:
        df[c] = df[c].fillna("").astype(str)
    if "Score" in df.columns:
        df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
    return df

# Numeric feature engineering
def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    summary = df.get("Summary", "").fillna("").astype(str).map(lambda x: clean_text(x, max_len=TEXT_TRUNCATE))
    text = df.get("Text", "").fillna("").astype(str).map(lambda x: clean_text(x, max_len=TEXT_TRUNCATE))

    df["summary_length"] = summary.str.len().astype("int32")
    df["text_length"] = text.str.len().astype("int32")
    df["summary_word_count"] = summary.str.split().str.len().astype("int32")
    df["text_word_count"] = text.str.split().str.len().astype("int32")

    # Simple sentiment and punctuation features
    both = (summary + " " + text).str.strip()
    df["positive_word_count"] = both.map(lambda x: count_vocab_words(x, POS_PATTERNS)).astype("int16")
    df["negative_word_count"] = both.map(lambda x: count_vocab_words(x, NEG_PATTERNS)).astype("int16")
    df["sentiment_score"] = (df["positive_word_count"] - df["negative_word_count"]).astype("int16")
    df["exclamation_count"] = both.str.count("!").astype("int16")
    df["question_count"] = both.str.count(r"\?").astype("int16")

    def caps_ratio_func(x: str) -> float:
        n = len(x)
        if n == 0:
            return 0.0
        up = sum(1 for c in x if c.isupper())
        r = up / n
        if not np.isfinite(r):
            r = 0.0
        return float(max(0.0, min(1.0, r)))

    df["caps_ratio"] = both.map(caps_ratio_func).astype("float32")

    # Log transforms
    log_cols = [
        "summary_length","text_length",
        "summary_word_count","text_word_count",
        "exclamation_count","question_count",
        "positive_word_count","negative_word_count",
    ]
    for col in log_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df[f"{col}_log"] = np.log1p(df[col].astype("float32")).astype("float32")

    # Save cleaned text for TF-IDF usage
    df["__clean_text__"] = both
    return df

def calculate_aggregate_stats(train_df: pd.DataFrame, use_entity_features: bool) -> Dict[str, Any]:
    train_df = train_df.copy()
    if "Score" in train_df.columns:
        train_df["Score"] = clean_score_column(train_df["Score"])

    overall_mean_score = float(pd.to_numeric(train_df["Score"], errors="coerce").dropna().mean()) if "Score" in train_df.columns else 3.0

    stats: Dict[str, Any] = {
        "overall_mean_score": overall_mean_score,
        "meta": {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "use_entity_features": use_entity_features,
            "random_state": RANDOM_STATE,
            "config_sig": CURRENT_SIG,  # 记录签名（新增）
        },
    }

    if use_entity_features:
        # Optional priors: historical means by ProductId/UserId
        stats["product_mean_scores"] = (
            train_df.dropna(subset=["Score"]).groupby("ProductId")["Score"].mean().astype(float).to_dict()
        )
        stats["user_mean_scores"] = (
            train_df.dropna(subset=["Score"]).groupby("UserId")["Score"].mean().astype(float).to_dict()
        )
        stats["product_counts"] = train_df["ProductId"].value_counts().astype(int).to_dict()
        stats["user_counts"] = train_df["UserId"].value_counts().astype(int).to_dict()

    return stats

def add_aggregate_features(df: pd.DataFrame, stats: Dict[str, Any], use_entity_features: bool) -> pd.DataFrame:
    df = df.copy()
    overall = stats["overall_mean_score"]

    if use_entity_features:
        df["product_mean_score"] = (
            df["ProductId"].map(stats.get("product_mean_scores", {})).fillna(overall).astype("float32")
        )
        df["user_mean_score"] = (
            df["UserId"].map(stats.get("user_mean_scores", {})).fillna(overall).astype("float32")
        )
        df["product_review_count"] = df["ProductId"].map(stats.get("product_counts", {})).fillna(1).astype("float32")
        df["user_review_count"] = df["UserId"].map(stats.get("user_counts", {})).fillna(1).astype("float32")
        for col in ["product_review_count","user_review_count"]:
            df[f"{col}_log"] = np.log1p(df[col]).astype("float32")
    return df

def select_numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {
        "Id","Score","ProductId","UserId","Summary","Text","__clean_text__",
    }
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if c not in exclude]

# Text vectorization utilities
def fit_text_vectorizer(text_series: pd.Series):
    if USE_HASHING_VECTORIZER:
        hv = HashingVectorizer(
            n_features=HASHING_N_FEATURES,
            alternate_sign=HASHING_ALTERNATE_SIGN,
            norm=None,
            lowercase=True,
            ngram_range=TFIDF_NGRAM_RANGE,
            token_pattern=r"(?u)\b\w+\b",
        )
        X_hash = hv.transform(text_series.values)
        tfidf_trans = TfidfTransformer(sublinear_tf=TFIDF_SUBLINEAR_TF)
        X_tfidf = tfidf_trans.fit_transform(X_hash)
        return ("hashing", hv, tfidf_trans, X_tfidf)
    else:
        tv = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            min_df=TFIDF_MIN_DF,
            max_df=TFIDF_MAX_DF,
            ngram_range=TFIDF_NGRAM_RANGE,
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
            sublinear_tf=TFIDF_SUBLINEAR_TF,
        )
        X_tfidf = tv.fit_transform(text_series.values)
        return ("tfidf", tv, None, X_tfidf)

def transform_text_vectorizer(fitted, text_series: pd.Series):
    mode, vec, tfidf_trans, _ = fitted
    if mode == "hashing":
        X_hash = vec.transform(text_series.values)
        X_tfidf = tfidf_trans.transform(X_hash)
        return X_tfidf
    else:
        X_tfidf = vec.transform(text_series.values)
        return X_tfidf

# Data loading and splitting
train_path = "./data/train.csv"
test_path = "./data/test.csv"

trainingSet = ensure_columns(safe_read_csv(train_path))
testingSet = ensure_columns(safe_read_csv(test_path))

print(f"Training set shape: {trainingSet.shape}  Test set shape: {testingSet.shape}")

#need to select those rows from train.csv as the test set
test_ids = testingSet["Id"].astype(str).values
trainingSet["Id"] = trainingSet["Id"].astype(str)

test_data = trainingSet[trainingSet["Id"].isin(test_ids)].copy()
train_data = trainingSet[~trainingSet["Id"].isin(test_ids)].copy()

# Keep only labeled training samples
train_data = train_data.copy()
train_data = train_data[train_data["Score"].notna()].copy()
train_data["Score"] = clean_score_column(train_data["Score"])

print("Score distribution (train):")
print(train_data["Score"].value_counts().sort_index())

# Group-based split by UserId to avoid user leakage
groups_for_split = train_data["UserId"].astype(str).values
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
train_idx, val_idx = next(gss.split(train_data, train_data["Score"], groups_for_split))
train_raw = train_data.iloc[train_idx].copy()
val_raw = train_data.iloc[val_idx].copy()

print(f"Train_raw: {train_raw.shape}, Val_raw: {val_raw.shape}")
overlap_users = set(train_raw["UserId"]).intersection(set(val_raw["UserId"]))
print(f"User overlap between train and val: {len(overlap_users)} (should be 0)")

# Numeric and aggregate features
train_basic = add_basic_features(train_raw)
val_basic = add_basic_features(val_raw)

# Load or compute aggregate priors
aggregate_stats = None
if os.path.exists(AGG_STATS_PKL):
    with open(AGG_STATS_PKL, "rb") as f:
        aggregate_stats = pickle.load(f)
    sig_in_stats = aggregate_stats.get("meta", {}).get("config_sig")
    if sig_in_stats != CURRENT_SIG:
        print("Aggregate stats config changed. Recomputing stats...")
        aggregate_stats = calculate_aggregate_stats(train_basic, use_entity_features=USE_ENTITY_AGG_FEATURES)
        with open(AGG_STATS_PKL, "wb") as f:
            pickle.dump(aggregate_stats, f)
    else:
        print("Loaded aggregate_stats from cache.")
else:
    aggregate_stats = calculate_aggregate_stats(
        train_basic, use_entity_features=USE_ENTITY_AGG_FEATURES
    )
    with open(AGG_STATS_PKL, "wb") as f:
        pickle.dump(aggregate_stats, f)
    print("Saved aggregate_stats to cache.")

train_full = add_aggregate_features(
    train_basic, aggregate_stats, use_entity_features=USE_ENTITY_AGG_FEATURES
)
val_full = add_aggregate_features(
    val_basic, aggregate_stats, use_entity_features=USE_ENTITY_AGG_FEATURES
)

feature_columns = select_numeric_feature_columns(train_full)
print(f"Using {len(feature_columns)} numeric features. Entity agg = {USE_ENTITY_AGG_FEATURES}")

# Numeric feature scaling
scaler = None
if os.path.exists(SCALER_PKL) and os.path.exists(FEATURE_COLUMNS_PKL):
    with open(SCALER_PKL, "rb") as f:
        scaler = pickle.load(f)
    with open(FEATURE_COLUMNS_PKL, "rb") as f:
        feature_columns_cached = pickle.load(f)
    if feature_columns_cached != feature_columns:
        print("Numeric feature columns changed. Refitting scaler...")
        scaler = StandardScaler().fit(
            train_full[feature_columns].fillna(0).values.astype("float32")
        )
        with open(SCALER_PKL, "wb") as f:
            pickle.dump(scaler, f)
        with open(FEATURE_COLUMNS_PKL, "wb") as f:
            pickle.dump(feature_columns, f)
    else:
        print("Loaded scaler and feature_columns from cache.")
else:
    scaler = StandardScaler().fit(
        train_full[feature_columns].fillna(0).values.astype("float32")
    )
    with open(SCALER_PKL, "wb") as f:
        pickle.dump(scaler, f)
    with open(FEATURE_COLUMNS_PKL, "wb") as f:
        pickle.dump(feature_columns, f)
    print("Saved scaler and feature_columns to cache.")

X_train_num = scaler.transform(
    train_full[feature_columns].fillna(0).values.astype("float32")
)
X_val_num = scaler.transform(
    val_full[feature_columns].fillna(0).values.astype("float32")
)

# Convert to sparse matrices
X_train_num_sp = sparse.csr_matrix(X_train_num)
X_val_num_sp = sparse.csr_matrix(X_val_num)

y_train = train_full["Score"].astype("int32").values
y_val = val_full["Score"].astype("int32").values

# Text TF-IDF features
train_text = train_full["__clean_text__"].fillna("")
val_text = val_full["__clean_text__"].fillna("")

fitted_text_vec = None
need_refit_text = True
if os.path.exists(TEXT_VECTORIZER_PKL):
    with open(TEXT_VECTORIZER_PKL, "rb") as f:
        tv_pack = pickle.load(f)
    # Check that config signature and vectorizer configuration are consistent
    pack_cfg = tv_pack.get("config", {})
    pack_sig = tv_pack.get("config_sig")
    expected_cfg = {
        "USE_HASHING_VECTORIZER": USE_HASHING_VECTORIZER,
        "TFIDF_MAX_FEATURES": TFIDF_MAX_FEATURES,
        "TFIDF_MIN_DF": TFIDF_MIN_DF,
        "TFIDF_MAX_DF": TFIDF_MAX_DF,
        "TFIDF_NGRAM_RANGE": TFIDF_NGRAM_RANGE,
        "HASHING_N_FEATURES": HASHING_N_FEATURES,
        "HASHING_ALTERNATE_SIGN": HASHING_ALTERNATE_SIGN,
        "TFIDF_SUBLINEAR_TF": TFIDF_SUBLINEAR_TF,
    }
    if (pack_cfg == expected_cfg) and (pack_sig == CURRENT_SIG):
        print("Loaded text vectorizer from cache.")
        mode = tv_pack["mode"]
        vec = tv_pack["vectorizer"]
        tfidf_trans = tv_pack["tfidf_transformer"]
        fitted_text_vec = (mode, vec, tfidf_trans, None)
        need_refit_text = False
    else:
        print("Text vectorizer config changed. Will refit.")

if need_refit_text:
    fitted_text_vec = fit_text_vectorizer(train_text)
    mode, vec, tfidf_trans, _ = fitted_text_vec
    with open(TEXT_VECTORIZER_PKL, "wb") as f:
        pickle.dump(
            {
                "mode": mode,
                "vectorizer": vec,
                "tfidf_transformer": tfidf_trans,
                "config": {
                    "USE_HASHING_VECTORIZER": USE_HASHING_VECTORIZER,
                    "TFIDF_MAX_FEATURES": TFIDF_MAX_FEATURES,
                    "TFIDF_MIN_DF": TFIDF_MIN_DF,
                    "TFIDF_MAX_DF": TFIDF_MAX_DF,
                    "TFIDF_NGRAM_RANGE": TFIDF_NGRAM_RANGE,
                    "HASHING_N_FEATURES": HASHING_N_FEATURES,
                    "HASHING_ALTERNATE_SIGN": HASHING_ALTERNATE_SIGN,
                    "TFIDF_SUBLINEAR_TF": TFIDF_SUBLINEAR_TF,
                },
                "config_sig": CURRENT_SIG,
            },
            f,
        )
    print("Saved text vectorizer to cache.")

X_train_txt = transform_text_vectorizer(fitted_text_vec, train_text)
X_val_txt = transform_text_vectorizer(fitted_text_vec, val_text)

print(f"Numeric features shape: {X_train_num_sp.shape}, Text TF-IDF shape: {X_train_txt.shape}")

# Concatenate numeric and text features
X_train_all = sparse.hstack([X_train_num_sp, X_train_txt], format="csr")
X_val_all = sparse.hstack([X_val_num_sp, X_val_txt], format="csr")
print(f"Combined train shape: {X_train_all.shape}, val shape: {X_val_all.shape}")

# Grouped cross-validation tuning
print("=" * 66)
print("FINE-TUNING WITH StratifiedGroupKFold on numeric+text (group=UserId)")
print("Metric: Accuracy")
print("=" * 66)

sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# Use externally fitted text space
X_all_num = scaler.transform(
    train_full[feature_columns].fillna(0).values.astype("float32")
)
X_all_num_sp = sparse.csr_matrix(X_all_num)
X_all_txt = transform_text_vectorizer(
    fitted_text_vec, train_full["__clean_text__"].fillna("")
)
X_all = sparse.hstack([X_all_num_sp, X_all_txt], format="csr")
y_all = train_full["Score"].astype("int32").values
g_all = train_full["UserId"].astype(str).values

best_C = None
best_mean = -1.0
cv_rows = []

for C_val in C_GRID:
    fold_scores = []
    for tr_idx, va_idx in sgkf.split(X_all, y_all, groups=g_all):
        X_tr = X_all[tr_idx]
        y_tr = y_all[tr_idx]
        X_va = X_all[va_idx]
        y_va = y_all[va_idx]

        if USE_MANUAL_CLASS_WEIGHT:
            cnt = Counter(y_tr)
            total = sum(cnt.values())
            n_classes = len(cnt)
            cw = {c: total / (n_classes * cnt[c]) for c in cnt}
            class_weight = cw
        else:
            class_weight = "balanced"

        clf = LogisticRegression(
            C=float(C_val),
            multi_class="multinomial",
            solver="lbfgs",
            class_weight=class_weight,
            max_iter=MODEL_MAX_ITER,
            random_state=RANDOM_STATE,
        )
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_va)
        acc = accuracy_score(y_va, y_pred)
        fold_scores.append(acc)

    mean_acc = float(np.mean(fold_scores))
    std_acc = float(np.std(fold_scores))
    cv_rows.append({"C": float(C_val), "mean_acc": mean_acc, "std_acc": std_acc, "scores": fold_scores})
    print(f"C={C_val:>8.6f}: GroupCV ACC = {mean_acc:.4f} (+/- {std_acc*2:.4f})  [{', '.join(f'{s:.3f}' for s in fold_scores)}]")

    if mean_acc > best_mean:
        best_mean = mean_acc
        best_C = float(C_val)

cv_results_df = pd.DataFrame(cv_rows)
with open(CV_RESULTS_PKL, "wb") as f:
    pickle.dump(cv_results_df, f)

print("-" * 60)
print(f"Best C: {best_C}  GroupCV Accuracy = {best_mean:.4f}")

# Final training and hold-out evaluation
# If there is a cached model with the same signature and C, reuse it; otherwise retrain
reuse_model = False
if os.path.exists(FINAL_MODEL_PKL):
    try:
        with open(FINAL_MODEL_PKL, "rb") as f:
            saved = pickle.load(f)
        model_loaded = saved.get("model")
        meta = saved.get("meta", {})
        if model_loaded is not None and meta.get("config_sig") == CURRENT_SIG and meta.get("best_C") == best_C:
            final_model = model_loaded
            reuse_model = True
            print("Loaded final model from cache.")
        else:
            print("Model cache found but config or C changed. Will retrain.")
    except Exception as e:
        print(f"Failed to load cached model: {e}. Will retrain.")

if not reuse_model:
    if USE_MANUAL_CLASS_WEIGHT:
        cnt_final = Counter(y_train)
        total_final = sum(cnt_final.values())
        n_classes_final = len(cnt_final)
        cw_final = {c: total_final / (n_classes_final * cnt_final[c]) for c in cnt_final}
        final_class_weight = cw_final
    else:
        final_class_weight = "balanced"

    final_model = LogisticRegression(
        C=best_C if best_C is not None else 1.0,
        multi_class="multinomial",
        solver="lbfgs",
        class_weight=final_class_weight,
        max_iter=MODEL_MAX_ITER,
        random_state=RANDOM_STATE,
    )
    final_model.fit(X_train_all, y_train)
    with open(FINAL_MODEL_PKL, "wb") as f:
        pickle.dump(
            {
                "model": final_model,
                "meta": {"config_sig": CURRENT_SIG, "best_C": best_C},
            },
            f,
        )
    print(f"Final model trained and saved. C={best_C}")

y_tr_pred = final_model.predict(X_train_all)
y_va_pred = final_model.predict(X_val_all)

print("=" * 50)
print("MODEL EVALUATION (Group-consistent, numeric+text)")
print("Metric: Accuracy")
print("=" * 50)
train_acc = accuracy_score(y_train, y_tr_pred)
val_acc = accuracy_score(y_val, y_va_pred)
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Val   Accuracy: {val_acc:.4f}")


# labels_sorted = sorted(pd.unique(train_data["Score"]))
# print("\nClassification Report (Val):")
# print(classification_report(y_val, y_va_pred, labels=labels_sorted, zero_division=0))
# print("\nConfusion Matrix (Val):")
# print(confusion_matrix(y_val, y_va_pred, labels=labels_sorted))

# Test set prediction
print("\nPreparing test set...")

test_df = test_data.copy()
# test_df already comes from the corresponding rows in train.csv, including Summary/Text and other info
# Re-ensure column completeness
test_df = ensure_columns(test_df)

# Numeric features
test_basic = add_basic_features(test_df)

# Load cached aggregate stats, columns, scaler, and text vectorizer
with open(AGG_STATS_PKL, "rb") as f:
    stats_loaded = pickle.load(f)
test_full = add_aggregate_features(
    test_basic, stats_loaded, use_entity_features=USE_ENTITY_AGG_FEATURES
)

with open(FEATURE_COLUMNS_PKL, "rb") as f:
    feature_columns_loaded = pickle.load(f)
X_test_num = (
    test_full.reindex(columns=feature_columns_loaded).fillna(0).astype("float32").values
)
with open(SCALER_PKL, "rb") as f:
    scaler_loaded = pickle.load(f)
X_test_num = scaler_loaded.transform(X_test_num)
X_test_num_sp = sparse.csr_matrix(X_test_num)

# Text features
with open(TEXT_VECTORIZER_PKL, "rb") as f:
    tv_pack = pickle.load(f)
mode = tv_pack["mode"]
vec = tv_pack["vectorizer"]
tfidf_trans = tv_pack["tfidf_transformer"]

def transform_text_for_test(mode, vec, tfidf_trans, series):
    if mode == "hashing":
        X_hash = vec.transform(series.values)
        return tfidf_trans.transform(X_hash)
    else:
        return vec.transform(series.values)

X_test_txt = transform_text_for_test(
    mode, vec, tfidf_trans, test_full["__clean_text__"].fillna("")
)

# Concatenate and predict
X_test_all = sparse.hstack([X_test_num_sp, X_test_txt], format="csr")

# Load final model
with open(FINAL_MODEL_PKL, "rb") as f:
    pack = pickle.load(f)
model_loaded = pack["model"]

test_pred = model_loaded.predict(X_test_all).astype(int)

# Generate submission file
submission = pd.DataFrame({"Id": test_full["Id"], "Score": test_pred}).sort_values("Id")
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")
print("Prediction distribution on test:")
print(submission["Score"].value_counts().sort_index())
print("Done.")