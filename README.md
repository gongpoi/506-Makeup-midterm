# CS506 Midterm

## 1. Data Exploration

- **Dataset structure**
  - Columns used: `Id`, `ProductId`, `UserId`, `Summary`, `Text`, and `Score` (target in train).
  - I first ensured that train and test had consistent schemas and converted all key IDs to string type for safety.

- **Cleaning and sanity checks**
  - `Score` was converted to numeric, clipped to the range 1-5, and rounded to integer.
  - Missing `Summary` / `Text` were imputed as empty strings so that vectorizers and feature functions never see `NaN`.
  - I verified the label distribution and found clear class imbalance (5-star reviews are much more frequent than low scores).

- **User-based split to avoid leakage**
  - The competition test set is defined via `Id` values that actually come from the same CSV as train.
  - I first split the original training data into:
    - A hold-out validation set via `GroupShuffleSplit` with `UserId` as the group key.
    - The remaining rows used as the main training set.
  - This ensures that no user appears in both train and validation, mimicking the "new user" scenario and preventing leakage through user behavior.

- **Initial patterns observed**
  - Longer reviews and summaries tend to correlate with extreme scores (1 or 5).
  - Very emotional text (many exclamation marks, strong positive/negative words, or lots of ALL CAPS) often corresponds to extreme ratings.
  - Some users consistently give higher or lower ratings than average, and some products get systematically lower scores.

---

## 2. Feature Extraction / Engineering

The final model uses a combination of engineered numeric features and text-based TF-IDF features, built in a reproducible pipeline.

### 2.1 Basic numeric features

From `Summary` and `Text` (after cleaning), I compute:

- **Length / count features**
  - `summary_length`, `text_length`
  - `summary_word_count`, `text_word_count`
- **Sentiment-related counts**
  - `positive_word_count` – occurrences of a small hand-crafted positive lexicon (e.g., “great”, “excellent”, “amazing”, …)
  - `negative_word_count` – occurrences of negative words (e.g., “bad”, “terrible”, “disappointed”, …)
  - `sentiment_score` = `positive_word_count - negative_word_count`
- **Punctuation & emphasis**
  - `exclamation_count` – number of `!`
  - `question_count` – number of `?`
  - `caps_ratio` – share of characters that are uppercase in the combined text, used as a proxy for “shouting” or excitement

To reduce skew, I also add **log-transformed versions** of many counts, e.g.:

- `summary_length_log`, `text_word_count_log`, `exclamation_count_log`, etc.

All these are numeric features that go into a `StandardScaler` before modeling.

### 2.2 Text cleaning and vectorization

- I construct a combined text field `__clean_text__ = clean(Summary) + " " + clean(Text)` where:
  - HTML tags are removed.
  - Excessive whitespace is collapsed.
  - Text is truncated to a maximum length (`TEXT_TRUNCATE`, e.g. 8000 chars) for efficiency.
- **Vectorization**
  - I use **TF-IDF** (`TfidfVectorizer`) with:
    - 1–2 grams (`ngram_range = (1, 2)`)
    - A cap on maximum number of features (`max_features = 200000`)
    - `min_df = 3`, `max_df = 0.98` to drop extremely rare or overly common tokens
    - `sublinear_tf = True` to dampen the effect of very frequent terms
  - There is also a configurable option to switch to a **HashingVectorizer + TfidfTransformer** pipeline for very large datasets, but the final submission uses the standard TF-IDF mode.

### 2.3 Optional aggregate features (implemented but off by default)

- The code supports entity-level aggregate features:
  - Mean historical score per `ProductId`
  - Mean historical score per `UserId`
  - Review counts per product and per user, with log transforms
- For entities unseen in training, these features fall back to the global mean score.
- For this submission, these aggregate features are disabled by a configuration flag to keep the pipeline simple and avoid introducing extra bias; they remain an avenue for future improvement.

### 2.4 Final feature matrix

- All numeric features are standardized with `StandardScaler`.
- Text features are TF-IDF vectors.
- I hstack the numeric and text features into a single sparse matrix:
  - `X = [scaled_numeric_features | text_tfidf_features]`

---

## 3. Model Creation and Assumptions

### 3.1 Choice of model

I use **multinomial Logistic Regression** as the core model:

- `multi_class="multinomial"` with the `lbfgs` solver.
- This model is efficient on high-dimensional sparse data and fits well with TF-IDF features.
- It is also easy to interpret and complies with the competition requirement to use classical methods (no deep learning / boosting).

### 3.2 Handling imbalance

- I set `class_weight="balanced"` so that rarer star ratings (e.g., 1-star, 2-star) are upweighted relative to 4- and 5-star reviews.
- There is also a code path for manual class weights (inverse-frequency based), but the default balanced mode works well and keeps the pipeline straightforward.

### 3.3 Assumptions

- The model assumes a **linear relationship** between features and the log-odds of each score class.
- Features are treated as largely **independent** inputs; TF-IDF helps reduce redundancy by downweighting overly common terms.
- For users/products without history (when aggregate features are enabled), the global mean is assumed to be a reasonable prior.

---

## 4. Model Tuning

Hyperparameter tuning focuses on the **regularization strength `C`** of Logistic Regression.

- **Search space**
  - I search over a **log-spaced grid** of C values (roughly from 0.01 to ~3.16), spanning strong to weak regularization.
- **Cross-validation strategy**
  - I use `StratifiedGroupKFold` with:
    - **Groups = `UserId`** to ensure each user appears in only one fold.
    - Stratification by `Score` to keep the label distribution similar across folds.
  - For each C, I compute mean and standard deviation of accuracy across the folds.
- **Selection criterion**
  - I select the C with the **highest mean CV accuracy** as the best trade-off between underfitting and overfitting.
- **Caching for reproducibility and speed**
  - To avoid re-fitting expensive components, I cache:
    - Numeric feature scaler
    - Text vectorizer (TF-IDF or hashing+TF-IDF)
    - Aggregate stats
    - Cross-validation results
    - Final trained model
  - A small **MD5 configuration signature** is used so that when any key configuration changes, the pipeline knows it must recompute the corresponding cached objects.

The final model is then retrained on the full training split and evaluated on the hold-out validation set.

---

## 5. Model Evaluation / Performance

- **Metrics**
  - Primary metric: **Accuracy**, matching the competition leaderboard.
  - The code also supports generating a classification report and confusion matrix (commented out by default to keep the Kaggle run lightweight).
- **Train vs. validation behavior**
  - Training accuracy is higher than validation accuracy, but the gap is moderate, indicating some overfitting but not catastrophic.
  - Most errors occur between **neighboring star levels**:
    - 4-star vs 5-star (very positive but slightly different intensity).
    - 1-star vs 2-star (strongly negative reviews with similar vocabulary).
- **Generalization & leakage control**
  - Because of:
    - User-based splitting (`GroupShuffleSplit` for hold-out, `StratifiedGroupKFold` for CV), and
    - A clear separation between model selection (CV) and final evaluation (hold-out),
  - The validation performance should be a **reasonable estimate** of how the model behaves on unseen users in the Kaggle test data.

---

## 6. Struggles / Issues / Open Questions

### 6.1 Struggles / issues

- **Class imbalance**
  - 5-star reviews dominate, leading the model to favor high scores.
  - `class_weight="balanced"` helps, but low-frequency classes (especially 1-star) still have lower recall.

- **High-dimensional text space**
  - TF-IDF with 1–2 grams at large vocabulary sizes creates a very high-dimensional feature space.
  - Even with strong regularization, there is a risk of the model memorizing noisy n-gram patterns.

- **Cold-start entities**
  - When enabling aggregate features (user/product mean scores), new or rarely seen users/products may not be well handled.
  - Using global averages is a simple fallback, but it may not fully capture user-specific biases.

- **Compute constraints**
  - Cross-validation over multiple C values on a large TF-IDF matrix is computationally expensive.
  - Caching mitigates this, but experimentation with very large n-gram spaces or additional models is still limited by runtime.

### 6.2 Open questions / future work

- **Richer models**
  - Could tree-based methods (e.g., Random Forests) or modern NLP models (e.g., BERT-style transformers) capture non-linear interaction patterns and subtle sentiment cues better, while still remaining within reasonable compute limits?
- **Better handling of imbalance**
  - Would focal loss-style ideas, custom resampling strategies, or per-class metric optimization significantly improve minority class performance?
- **Dimensionality reduction / feature selection**
  - Applying techniques like PCA, truncated SVD (LSA), or feature selection on TF-IDF might reduce noise and improve generalization.
- **Richer interaction features**
  - Features combining length, sentiment, and punctuation could help separate ambiguous cases that currently end up in neighboring classes.
