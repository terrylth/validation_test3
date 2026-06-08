"""
Vendor Dispute Escalation Classifier
SMBC Data Management Office
-------------------------------------
Purpose:
    Binary classification model to predict whether a procurement or vendor
    dispute will escalate to legal counsel or senior management review.

Target variable:
    escalated (1 = escalated, 0 = resolved at operational level)

Features:
    - Structured: contract_value, dispute_type, vendor_tier, days_outstanding,
                  prior_disputes, resolution_attempts, region, dispute_month
    - Text: dispute_notes
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    precision_score, recall_score, f1_score, brier_score_loss
)
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)


# ─────────────────────────────────────────────
# DATA GENERATOR
# ─────────────────────────────────────────────

class DataGenerator:
    """
    Generates synthetic vendor dispute data with structured
    and text features ordered chronologically.
    """

    DISPUTE_TYPES = ["payment_delay", "quality_issue", "contract_breach",
                     "delivery_failure", "scope_dispute", "ip_violation"]
    REGIONS       = ["APAC", "EMEA", "Americas", "SEA", "Japan"]
    VENDOR_TIERS  = [1, 2, 3]

    def __init__(self, n: int = 4000, seed: int = 42):
        self.n    = n
        self.seed = seed
        np.random.seed(seed)

    def generate(self) -> pd.DataFrame:
        base_date = pd.Timestamp("2022-01-01")
        n         = self.n

        df = pd.DataFrame({
            "dispute_id":          [f"DSP_{i:05d}" for i in range(n)],
            "dispute_date":        [base_date + pd.Timedelta(days=int(i * 730 / n))
                                    for i in range(n)],
            "dispute_type":        np.random.choice(self.DISPUTE_TYPES, n),
            "vendor_tier":         np.random.choice(self.VENDOR_TIERS, n, p=[0.5, 0.35, 0.15]),
            "region":              np.random.choice(self.REGIONS, n),
            "days_outstanding":    np.random.randint(1, 180, n),
            "prior_disputes":      np.random.poisson(lam=1.2, size=n),
            "resolution_attempts": np.random.poisson(lam=2.0, size=n),
            "contract_value":      np.random.lognormal(mean=10, sigma=1.5, size=n).round(2),
            "dispute_month":       [(base_date + pd.Timedelta(days=int(i * 730 / n))).month
                                    for i in range(n)],
        })

        escalation_prob = (
            0.06
            + 0.10 * (df["vendor_tier"] == 3).astype(int)
            + 0.08 * (df["days_outstanding"] > 60).astype(int)
            + 0.09 * (df["prior_disputes"] >= 3).astype(int)
            + 0.07 * (df["dispute_type"] == "ip_violation").astype(int)
            + 0.06 * (df["dispute_type"] == "contract_breach").astype(int)
            + 0.05 * (df["resolution_attempts"] >= 4).astype(int)
            + 0.04 * (np.log1p(df["contract_value"]) > 13).astype(int)
        ).clip(0, 1)

        df["escalated"] = (np.random.rand(n) < escalation_prob).astype(int)

        # Introduce duplicate rows
        duplicate_idx = np.random.choice(n, size=int(n * 0.03), replace=False)
        duplicates    = df.iloc[duplicate_idx].copy()
        df            = pd.concat([df, duplicates], ignore_index=True)
        df            = df.sample(frac=1, random_state=42).reset_index(drop=True)

        low_urgency = [
            "Standard payment terms dispute requiring clarification",
            "Minor delivery schedule discrepancy noted in contract",
            "Quality metrics slightly below agreed thresholds",
            "Invoice reconciliation issue under review",
            "Routine scope clarification with vendor requested",
            "Vendor requested extension on delivery milestone",
        ]
        high_urgency = [
            "Vendor has materially breached core contract obligations",
            "Significant IP violation identified requiring legal review",
            "Critical delivery failure impacting business operations",
            "Vendor unresponsive to multiple escalation attempts",
            "Substantial financial exposure due to contract non-compliance",
            "Repeated quality failures despite formal corrective action plan",
        ]

        descriptions = []
        for esc in df["escalated"]:
            base  = np.random.choice(high_urgency if esc == 1 else low_urgency)
            noise = np.random.choice(["", " requires attention", " urgent review needed", ""],
                                     p=[0.5, 0.2, 0.1, 0.2])
            descriptions.append(base + noise)

        df["dispute_notes"] = descriptions
        return df


# ─────────────────────────────────────────────
# FEATURE ENGINEER
# ─────────────────────────────────────────────

class FeatureEngineer:
    """
    Constructs structured features from raw dispute data.
    """

    def __init__(self, df_full: pd.DataFrame):
        self._df_full = df_full

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().reset_index(drop=True)

        df["log_contract_value"] = np.log1p(df["contract_value"])
        df["high_value_dispute"] = (df["contract_value"] > 100000).astype(int)
        df["repeat_offender"]    = (df["prior_disputes"] >= 3).astype(int)
        df["long_outstanding"]   = (df["days_outstanding"] > 60).astype(int)

        # Interaction feature
        df["value_days_interaction"] = (
            np.log1p(df["contract_value"]) * df["days_outstanding"]
        ) / self._df_full["days_outstanding"].mean()

        # Rolling dispute rate per vendor tier over prior 30 days
        df_sorted  = self._df_full.sort_values("dispute_date").reset_index(drop=True)
        tier_rates = {}
        for idx, row in df_sorted.iterrows():
            window = df_sorted[
                (df_sorted["dispute_date"] >= row["dispute_date"] - pd.Timedelta(days=30)) &
                (df_sorted["dispute_date"] <= row["dispute_date"]) &
                (df_sorted["vendor_tier"] == row["vendor_tier"])
            ]
            tier_rates[idx] = window["escalated"].mean()

        rate_series = pd.Series(tier_rates)
        df["vendor_tier_recent_rate"] = df.index.map(
            lambda i: rate_series.get(i, np.nan)
        ).fillna(rate_series.mean())

        # Region escalation rate
        region_rate = self._df_full.groupby("region")["escalated"].mean()
        df["region_esc_rate"] = df["region"].map(region_rate)

        # Label encode categoricals
        le = LabelEncoder()
        for col in ["dispute_type", "region"]:
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))

        return df


# ─────────────────────────────────────────────
# TEXT PROCESSOR
# ─────────────────────────────────────────────

class TextProcessor:
    """
    Extracts TF-IDF features from dispute notes.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=50,
            ngram_range=(1, 2),
            min_df=2,
            stop_words="english"
        )

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        matrix = self.vectorizer.fit_transform(df["dispute_notes"])
        return pd.DataFrame(
            matrix.toarray(),
            columns=[f"tfidf_{i}" for i in range(matrix.shape[1])],
            index=df.index
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        matrix = self.vectorizer.transform(df["dispute_notes"])
        return pd.DataFrame(
            matrix.toarray(),
            columns=[f"tfidf_{i}" for i in range(matrix.shape[1])],
            index=df.index
        )


# ─────────────────────────────────────────────
# PREPROCESSOR
# ─────────────────────────────────────────────

class Preprocessor:
    """
    Orchestrates feature engineering, text extraction,
    scaling, and train/test splitting.
    """

    STRUCTURED_COLS = [
        "log_contract_value", "high_value_dispute", "repeat_offender",
        "long_outstanding", "value_days_interaction", "vendor_tier_recent_rate",
        "region_esc_rate", "days_outstanding", "prior_disputes",
        "resolution_attempts", "vendor_tier", "dispute_month",
        "dispute_type_enc", "region_enc"
    ]

    def __init__(self, df: pd.DataFrame):
        self._df           = df
        self.feature_eng   = FeatureEngineer(df_full=df)
        self.text_proc     = TextProcessor()
        self.scaler        = StandardScaler()

    def run(self):
        df_eng   = self.feature_eng.transform(self._df)
        text_df  = self.text_proc.fit_transform(df_eng)

        X = pd.concat(
            [df_eng[self.STRUCTURED_COLS].reset_index(drop=True),
             text_df.reset_index(drop=True)],
            axis=1
        )
        y = df_eng["escalated"].reset_index(drop=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_train_sc = pd.DataFrame(
            self.scaler.fit_transform(X_train), columns=X.columns
        )
        X_test_sc  = pd.DataFrame(
            self.scaler.transform(X_test), columns=X.columns
        )

        return (X_train_sc, X_test_sc,
                y_train.reset_index(drop=True),
                y_test.reset_index(drop=True),
                X, y)


# ─────────────────────────────────────────────
# CROSS VALIDATOR
# ─────────────────────────────────────────────

class CrossValidator:
    """
    Evaluates model generalisation using stratified k-fold
    cross-validation.
    """

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    def run(self, model, X_train, y_train, X_full, y_full) -> np.ndarray:
        cv = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=42
        )
        scores = cross_val_score(
            model, X_full, y_full, cv=cv, scoring="roc_auc"
        )

        print("\n[Cross Validation]")
        print(f"  CV ROC-AUC scores : {np.round(scores, 4)}")
        print(f"  Mean CV ROC-AUC   : {scores.mean():.4f} (+/- {scores.std():.4f})")
        return scores


# ─────────────────────────────────────────────
# HYPERPARAMETER TUNER
# ─────────────────────────────────────────────

class HyperparameterTuner:
    """
    Selects optimal learning rate via grid search.
    """

    LEARNING_RATES = [0.01, 0.05, 0.10, 0.15, 0.20]

    def tune(self, X_train, X_test, y_train, y_test) -> float:
        best_auc = 0
        best_lr  = None

        print("\n[Hyperparameter Tuning]")
        for lr in self.LEARNING_RATES:
            m = GradientBoostingClassifier(
                n_estimators=100, learning_rate=lr,
                max_depth=3, random_state=42
            )
            m.fit(X_train, y_train)
            auc = roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])
            print(f"  learning_rate={lr:.2f}  Test AUC={auc:.4f}")
            if auc > best_auc:
                best_auc = auc
                best_lr  = lr

        print(f"  Selected learning_rate : {best_lr}")
        return best_lr


# ─────────────────────────────────────────────
# MODEL TRAINER
# ─────────────────────────────────────────────

class ModelTrainer:
    """
    Trains and evaluates the GradientBoosting escalation classifier.
    Applies probability calibration for reliable score outputs.
    """

    def __init__(self, learning_rate: float = 0.05):
        self.learning_rate = learning_rate
        self.model         = None

    def train(self, X_train, y_train) -> None:
        base_model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=self.learning_rate,
            max_depth=3,
            random_state=42
        )
        self.model = CalibratedClassifierCV(
            base_model, method="isotonic", cv=3
        )
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test) -> np.ndarray:
        y_pred  = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        print("\n[Model Evaluation - Test Set]")
        print(f"  Accuracy    : {accuracy_score(y_test, y_pred):.4f}")
        print(f"  ROC-AUC     : {roc_auc_score(y_test, y_proba):.4f}")
        print(f"  Precision   : {precision_score(y_test, y_pred):.4f}")
        print(f"  Recall      : {recall_score(y_test, y_pred):.4f}")
        print(f"  F1 Score    : {f1_score(y_test, y_pred):.4f}")
        print(f"  Brier Score : {brier_score_loss(y_test, y_proba):.4f}")
        print("\n  Classification Report:")
        print(classification_report(y_test, y_pred))

        return y_proba

    def feature_importance(self, feature_names: list, top_n: int = 15) -> None:
        base = self.model.calibrated_classifiers_[0].estimator
        imps = pd.Series(base.feature_importances_, index=feature_names)
        top  = imps.sort_values(ascending=False).head(top_n)

        print(f"\n[Top {top_n} Feature Importances]")
        for feat, imp in top.items():
            print(f"  {feat:<40} {imp:.4f}")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

class EscalationPipeline:
    """
    End-to-end pipeline for training and evaluating the
    vendor dispute escalation classifier.
    """

    def __init__(self, n: int = 4000):
        self.n = n

    def run(self):
        print("=" * 60)
        print("  Vendor Dispute Escalation Classifier — Training")
        print("=" * 60)

        print("\n[1] Generating data...")
        generator = DataGenerator(n=self.n)
        df        = generator.generate()
        print(f"    Dataset shape   : {df.shape}")
        print(f"    Escalation rate : {df['escalated'].mean():.2%}")

        print("\n[2] Preprocessing...")
        preprocessor = Preprocessor(df)
        X_train, X_test, y_train, y_test, X_full, y_full = preprocessor.run()
        print(f"    Train size      : {X_train.shape}")
        print(f"    Test size       : {X_test.shape}")

        print("\n[3] Cross-validation...")
        base_model = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=42
        )
        cv = CrossValidator(n_splits=5)
        cv.run(base_model, X_train, y_train, X_full, y_full)

        print("\n[4] Hyperparameter tuning...")
        tuner   = HyperparameterTuner()
        best_lr = tuner.tune(X_train, X_test, y_train, y_test)

        print("\n[5] Training final model...")
        trainer = ModelTrainer(learning_rate=best_lr)
        trainer.train(X_train, y_train)
        trainer.evaluate(X_test, y_test)

        print("\n[6] Feature importances...")
        trainer.feature_importance(X_train.columns.tolist())

        print("\n" + "=" * 60)
        print("  Training complete.")
        print("=" * 60)

        return trainer.model, X_train, X_test, y_train, y_test, X_full, y_full, df, preprocessor


if __name__ == "__main__":
    pipeline = EscalationPipeline(n=4000)
    pipeline.run()
