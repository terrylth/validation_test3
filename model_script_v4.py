"""
Internal Audit Finding Escalation Classifier
SMBC Data Management Office
----------------------------------------------
Purpose:
    Binary classification model to predict whether an internal audit
    finding will be escalated to the Board Risk Committee for review.

Target variable:
    escalated (1 = escalated to Board Risk Committee, 0 = resolved operationally)

Features:
    - Structured: finding_severity, business_unit, audit_cycle, finding_age,
                  auditor_grade, remediation_attempts, prior_findings,
                  days_since_last_audit, region
    - Text: finding_description
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)


# ─────────────────────────────────────────────
# DATA GENERATOR
# ─────────────────────────────────────────────

class DataGenerator:
    """
    Generates synthetic internal audit finding data with
    structured and text features ordered chronologically.
    """

    BUSINESS_UNITS = ["Treasury", "Compliance", "Operations",
                      "Technology", "Finance", "Risk Management",
                      "Legal", "HR"]
    REGIONS        = ["APAC", "EMEA", "Americas", "SEA", "Japan"]
    SEVERITIES     = ["low", "medium", "high", "critical"]
    AUDIT_CYCLES   = ["Q1", "Q2", "Q3", "Q4"]

    def __init__(self, n: int = 4000, seed: int = 42):
        self.n    = n
        self.seed = seed
        np.random.seed(seed)

    def generate(self) -> pd.DataFrame:
        n         = self.n
        base_date = pd.Timestamp("2021-01-01")

        df = pd.DataFrame({
            "finding_id":            [f"AUD_{i:05d}" for i in range(n)],
            "finding_date":          [base_date + pd.Timedelta(days=int(i * 730 / n))
                                      for i in range(n)],
            "business_unit":         np.random.choice(self.BUSINESS_UNITS, n),
            "region":                np.random.choice(self.REGIONS, n),
            "finding_severity":      np.random.choice(
                                         self.SEVERITIES, n,
                                         p=[0.30, 0.35, 0.25, 0.10]
                                     ),
            "audit_cycle":           np.random.choice(self.AUDIT_CYCLES, n),
            "auditor_grade":         np.random.choice([1, 2, 3, 4], n,
                                                      p=[0.4, 0.3, 0.2, 0.1]),
            "finding_age":           np.random.randint(1, 365, n),
            "remediation_attempts":  np.random.poisson(lam=2.5, size=n),
            "prior_findings":        np.random.poisson(lam=1.8, size=n),
            "days_since_last_audit": np.random.randint(30, 540, n),
        })

        severity_map = {"low": 0.04, "medium": 0.09,
                        "high": 0.17, "critical": 0.28}
        escalation_prob = (
            0.05
            + df["finding_severity"].map(severity_map)
            + 0.08 * (df["finding_age"] > 180).astype(int)
            + 0.07 * (df["prior_findings"] >= 3).astype(int)
            + 0.06 * (df["remediation_attempts"] >= 4).astype(int)
            + 0.05 * (df["auditor_grade"] >= 3).astype(int)
            + 0.04 * (df["days_since_last_audit"] > 365).astype(int)
        ).clip(0, 1)

        df["escalated"] = (np.random.rand(n) < escalation_prob).astype(int)

        df["days_to_remediation"] = (
            np.random.randint(10, 90, n)
            + df["escalated"] * np.random.randint(30, 180, n)
        )

        df["board_review_count"] = (
            np.random.poisson(lam=0.5, size=n)
            + df["escalated"] * np.random.poisson(lam=3.0, size=n)
        )

        low_urgency = [
            "Minor documentation gap identified in standard operating procedure",
            "Routine control deficiency noted in reconciliation process",
            "Low risk process deviation observed during sample testing",
            "Administrative oversight identified in reporting template",
            "Minor data quality issue noted in transaction records",
            "Standard policy compliance gap identified in onboarding process",
        ]
        high_urgency = [
            "Critical control failure identified with significant regulatory exposure",
            "Material breach of risk appetite framework requiring immediate remediation",
            "Systemic process breakdown identified across multiple business lines",
            "Significant governance gap identified in board reporting framework",
            "Repeated control failures despite prior management action plans",
            "High-risk finding with potential regulatory reporting implications",
        ]

        descriptions = []
        for esc in df["escalated"]:
            base  = np.random.choice(high_urgency if esc == 1 else low_urgency)
            noise = np.random.choice(
                ["", " requires immediate attention",
                 " flagged for senior review", ""],
                p=[0.5, 0.2, 0.1, 0.2]
            )
            descriptions.append(base + noise)

        df["finding_description"] = descriptions
        return df


# ─────────────────────────────────────────────
# FEATURE ENGINEER
# ─────────────────────────────────────────────

class FeatureEngineer:
    """
    Constructs structured features from raw audit finding data.
    """

    SEVERITY_ORDER = ["low", "medium", "high", "critical"]

    def __init__(self, df_full: pd.DataFrame):
        self._df_full = df_full.sort_values(
            "finding_date"
        ).reset_index(drop=True)

    def _compute_rolling_remediation_rate(
        self, df: pd.DataFrame
    ) -> pd.Series:
        df_sorted = self._df_full.copy()
        rates     = []
        for idx, row in df_sorted.iterrows():
            window = df_sorted[
                (df_sorted["finding_date"] >= row["finding_date"]
                 - pd.Timedelta(days=90)) &
                (df_sorted["finding_date"] <= row["finding_date"]) &
                (df_sorted["business_unit"] == row["business_unit"])
            ]
            rate = (window["remediation_attempts"] >= 3).mean()
            rates.append(rate)

        rate_series = pd.Series(rates, index=df_sorted.index)
        df_indexed  = df.set_index(
            df_sorted.index[:len(df)]
        ) if len(df) == len(df_sorted) else df
        return rate_series

    def _target_encode_business_unit(self, df: pd.DataFrame) -> pd.Series:
        encoding_map = self._df_full.groupby(
            "business_unit"
        )["escalated"].mean()
        return df["business_unit"].map(encoding_map)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().reset_index(drop=True)

        severity_enc = LabelEncoder()
        severity_enc.fit(self.SEVERITY_ORDER)
        df["severity_enc"] = severity_enc.transform(df["finding_severity"])

        df["high_severity"]       = df["finding_severity"].isin(
            ["high", "critical"]
        ).astype(int)
        df["long_outstanding"]    = (df["finding_age"] > 180).astype(int)
        df["repeat_unit"]         = (df["prior_findings"] >= 3).astype(int)
        df["high_remediation"]    = (df["remediation_attempts"] >= 4).astype(int)
        df["overdue_audit"]       = (df["days_since_last_audit"] > 365).astype(int)

        df["bu_escalation_rate"]  = self._target_encode_business_unit(df)

        rolling_rates             = self._compute_rolling_remediation_rate(df)
        df["rolling_remediation"] = rolling_rates.values[:len(df)]

        df["post_event_interaction"] = (
            df["days_to_remediation"] * df["board_review_count"]
        )

        region_dummies = pd.get_dummies(df["region"], prefix="region")
        cycle_dummies  = pd.get_dummies(df["audit_cycle"], prefix="cycle")
        df = pd.concat([df, region_dummies, cycle_dummies], axis=1)

        le = LabelEncoder()
        df["business_unit_enc"] = le.fit_transform(df["business_unit"])

        return df


# ─────────────────────────────────────────────
# TEXT PROCESSOR
# ─────────────────────────────────────────────

class TextProcessor:
    """
    Extracts TF-IDF features from audit finding descriptions.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=60,
            ngram_range=(1, 2),
            min_df=2
        )
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        matrix       = self.vectorizer.fit_transform(df["finding_description"])
        self._fitted = True
        return pd.DataFrame(
            matrix.toarray(),
            columns=[f"tfidf_{i}" for i in range(matrix.shape[1])],
            index=df.index
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("TextProcessor must be fit before transform.")
        matrix = self.vectorizer.transform(df["finding_description"])
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
        "severity_enc", "high_severity", "long_outstanding",
        "repeat_unit", "high_remediation", "overdue_audit",
        "bu_escalation_rate", "rolling_remediation",
        "post_event_interaction", "finding_age",
        "remediation_attempts", "prior_findings",
        "days_since_last_audit", "auditor_grade",
        "business_unit_enc"
    ]

    def __init__(self, df: pd.DataFrame):
        self._df         = df
        self.feature_eng = FeatureEngineer(df_full=df)
        self.text_proc   = TextProcessor()
        self.scaler      = StandardScaler()

    def run(self):
        df_eng  = self.feature_eng.transform(self._df)
        text_df = self.text_proc.fit_transform(df_eng)

        region_cols = [c for c in df_eng.columns if c.startswith("region_")]
        cycle_cols  = [c for c in df_eng.columns if c.startswith("cycle_")]
        all_cols    = self.STRUCTURED_COLS + region_cols + cycle_cols

        X = pd.concat(
            [df_eng[all_cols].reset_index(drop=True),
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

        return (
            X_train_sc, X_test_sc,
            y_train.reset_index(drop=True),
            y_test.reset_index(drop=True),
            X, y
        )


# ─────────────────────────────────────────────
# CROSS VALIDATOR
# ─────────────────────────────────────────────

class CrossValidator:
    """
    Evaluates model generalisation using walk-forward
    cross-validation to respect temporal ordering.
    """

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    def run(self, model, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        n        = len(X)
        fold_size = n // (self.n_splits + 1)
        scores   = []

        print("\n[Cross Validation — Walk Forward]")
        for i in range(self.n_splits):
            test_start  = (i + 1) * fold_size
            test_end    = (i + 2) * fold_size
            train_end   = test_start

            X_tr = X.iloc[:train_end]
            y_tr = y.iloc[:train_end]
            X_va = X.iloc[test_start:test_end]
            y_va = y.iloc[test_start:test_end]

            scaler   = StandardScaler()
            X_tr_sc  = pd.DataFrame(
                scaler.fit_transform(X_tr), columns=X.columns
            )
            X_va_sc  = pd.DataFrame(
                scaler.transform(X_va), columns=X.columns
            )

            m = RandomForestClassifier(
                n_estimators=100, random_state=42
            )
            m.fit(X_tr_sc, y_tr)
            auc = roc_auc_score(y_va, m.predict_proba(X_va_sc)[:, 1])
            scores.append(auc)
            print(f"  Fold {i+1}  train=[0:{train_end}]  "
                  f"val=[{test_start}:{test_end}]  AUC={auc:.4f}")

        scores = np.array(scores)
        print(f"  Mean CV AUC : {scores.mean():.4f} "
              f"(+/- {scores.std():.4f})")
        return scores


# ─────────────────────────────────────────────
# HYPERPARAMETER TUNER
# ─────────────────────────────────────────────

class HyperparameterTuner:
    """
    Selects optimal n_estimators using a validation set.
    """

    ESTIMATOR_CANDIDATES = [50, 100, 150, 200, 250]

    def tune(
        self, X_train, X_val, y_train, y_val
    ) -> int:
        best_auc  = 0
        best_n    = None

        print("\n[Hyperparameter Tuning]")
        for n_est in self.ESTIMATOR_CANDIDATES:
            m = RandomForestClassifier(
                n_estimators=n_est, random_state=42
            )
            m.fit(X_train, y_train)
            auc = roc_auc_score(
                y_val, m.predict_proba(X_val)[:, 1]
            )
            print(f"  n_estimators={n_est:<5}  Val AUC={auc:.4f}")
            if auc > best_auc:
                best_auc = auc
                best_n   = n_est

        print(f"  Selected n_estimators : {best_n}")
        return best_n


# ─────────────────────────────────────────────
# MODEL TRAINER
# ─────────────────────────────────────────────

class ModelTrainer:
    """
    Trains and evaluates the RandomForest escalation classifier.
    """

    def __init__(self, n_estimators: int = 100):
        self.n_estimators = n_estimators
        self.model        = None

    def train(self, X_train, y_train) -> None:
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42
        )
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test) -> np.ndarray:
        y_pred  = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        print("\n[Model Evaluation - Test Set]")
        print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
        print(f"  ROC-AUC   : {roc_auc_score(y_test, y_proba):.4f}")
        print(f"  Precision : {precision_score(y_test, y_pred):.4f}")
        print(f"  Recall    : {recall_score(y_test, y_pred):.4f}")
        print(f"  F1 Score  : {f1_score(y_test, y_pred):.4f}")
        print("\n  Classification Report:")
        print(classification_report(y_test, y_pred))

        return y_proba

    def feature_importance(
        self, feature_names: list, top_n: int = 15
    ) -> None:
        imps = pd.Series(
            self.model.feature_importances_, index=feature_names
        )
        top  = imps.sort_values(ascending=False).head(top_n)

        print(f"\n[Top {top_n} Feature Importances]")
        for feat, imp in top.items():
            print(f"  {feat:<40} {imp:.4f}")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

class AuditEscalationPipeline:
    """
    End-to-end pipeline for training and evaluating the
    internal audit finding escalation classifier.
    """

    def __init__(self, n: int = 4000):
        self.n = n

    def run(self):
        print("=" * 60)
        print("  Audit Finding Escalation Classifier — Training")
        print("=" * 60)

        print("\n[1] Generating data...")
        generator = DataGenerator(n=self.n)
        df        = generator.generate()
        print(f"    Dataset shape   : {df.shape}")
        print(f"    Escalation rate : {df['escalated'].mean():.2%}")

        print("\n[2] Preprocessing...")
        preprocessor = Preprocessor(df)
        (X_train, X_test,
         y_train, y_test,
         X_full, y_full) = preprocessor.run()
        print(f"    Train size      : {X_train.shape}")
        print(f"    Test size       : {X_test.shape}")

        print("\n[3] Cross-validation...")
        base_model = RandomForestClassifier(
            n_estimators=100, random_state=42
        )
        cv = CrossValidator(n_splits=5)
        cv.run(base_model, X_full, y_full)

        print("\n[4] Hyperparameter tuning...")
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2,
            random_state=42, stratify=y_train
        )
        tuner  = HyperparameterTuner()
        best_n = tuner.tune(X_tr, X_val, y_tr, y_val)

        print("\n[5] Training final model...")
        trainer = ModelTrainer(n_estimators=best_n)
        trainer.train(X_train, y_train)
        trainer.evaluate(X_test, y_test)

        print("\n[6] Feature importances...")
        trainer.feature_importance(X_train.columns.tolist())

        print("\n" + "=" * 60)
        print("  Training complete.")
        print("=" * 60)

        return (trainer.model, X_train, X_test,
                y_train, y_test, X_full, y_full,
                df, preprocessor)


if __name__ == "__main__":
    pipeline = AuditEscalationPipeline(n=4000)
    pipeline.run()
