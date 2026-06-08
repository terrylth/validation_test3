
"""
flawed_testing_script.py

Interview practice scenario:
Validation/testing script for escalation classification model.

Candidate task:
Check if the validation tests are correctly implemented.
Focus on:
- PSI implementation
- stability testing
- robustness testing
- sensitivity testing
- whether the tests really test what they claim
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


class ModelValidationTests:
    def __init__(self, model_pipeline):
        self.model_pipeline = model_pipeline

    def calculate_psi(self, expected, actual, buckets=10):
        """
        FLAWED PSI implementation.

        Intended formula:
        PSI = sum((actual% - expected%) * ln(actual% / expected%))

        Candidate should check:
        - binning
        - use of expected vs actual
        - zero handling
        - whether bins are based on expected distribution
        - whether direction matters
        """
        expected = pd.Series(expected).dropna()
        actual = pd.Series(actual).dropna()

        # FLAW: bins are based on combined data, not expected/reference data.
        combined = pd.concat([expected, actual], axis=0)
        cuts = np.percentile(combined, np.linspace(0, 100, buckets + 1))
        cuts = np.unique(cuts)

        expected_counts = pd.cut(expected, bins=cuts, include_lowest=True).value_counts(normalize=True)
        actual_counts = pd.cut(actual, bins=cuts, include_lowest=True).value_counts(normalize=True)

        psi_df = pd.DataFrame({
            "expected_pct": expected_counts,
            "actual_pct": actual_counts
        }).fillna(0.0001)

        # FLAW: formula direction is reversed versus the documented convention.
        # Magnitude may be similar but interpretation and audit defensibility suffer.
        psi_df["psi"] = (
            (psi_df["expected_pct"] - psi_df["actual_pct"])
            * np.log(psi_df["expected_pct"] / psi_df["actual_pct"])
        )

        return psi_df["psi"].sum()

    def psi_feature_report(self, train_df, prod_df, features):
        report = []
        for feature in features:
            # FLAW: applies numeric PSI to all features, including categorical/text if passed.
            value = self.calculate_psi(train_df[feature], prod_df[feature])
            status = "PASS" if value < 0.25 else "FAIL"
            report.append({"feature": feature, "psi": value, "status": status})
        return pd.DataFrame(report)

    def prediction_stability_test(self, X_reference, X_challenger):
        """
        Intended:
        Check whether prediction distribution is stable across reference and challenger periods.

        FLAW:
        Only compares mean predicted probability. Two distributions can have same mean
        but very different shape.
        """
        ref_pred = self.model_pipeline.predict_proba(X_reference)[:, 1]
        ch_pred = self.model_pipeline.predict_proba(X_challenger)[:, 1]

        mean_diff = abs(ref_pred.mean() - ch_pred.mean())
        return {
            "reference_mean_score": ref_pred.mean(),
            "challenger_mean_score": ch_pred.mean(),
            "mean_diff": mean_diff,
            "status": "PASS" if mean_diff < 0.05 else "FAIL"
        }

    def robustness_noise_test(self, X_test, y_test, numeric_features, noise_level=0.01):
        """
        Intended:
        Perturb numeric features and check performance degradation.

        FLAWS:
        - Adds tiny absolute noise, not scaled to feature distributions.
        - Mutates all numeric features including binary flags.
        - Does not repeat over random seeds.
        - Does not compare confidence intervals.
        """
        X_noisy = X_test.copy()
        rng = np.random.default_rng(123)

        for col in numeric_features:
            X_noisy[col] = X_noisy[col] + rng.normal(0, noise_level, size=len(X_noisy))

        base_pred = self.model_pipeline.predict_proba(X_test)[:, 1]
        noisy_pred = self.model_pipeline.predict_proba(X_noisy)[:, 1]

        base_auc = roc_auc_score(y_test, base_pred)
        noisy_auc = roc_auc_score(y_test, noisy_pred)

        return {
            "base_auc": base_auc,
            "noisy_auc": noisy_auc,
            "auc_drop": base_auc - noisy_auc,
            "status": "PASS" if (base_auc - noisy_auc) < 0.03 else "FAIL"
        }

    def feature_removal_sensitivity_test(self, X_test, y_test, feature_to_remove):
        """
        Intended:
        Test sensitivity to removing one feature.

        FLAW:
        Replaces the feature with zero without checking whether zero is meaningful.
        For categorical/text columns this may be invalid.
        """
        X_changed = X_test.copy()
        X_changed[feature_to_remove] = 0

        base_pred = self.model_pipeline.predict_proba(X_test)[:, 1]
        changed_pred = self.model_pipeline.predict_proba(X_changed)[:, 1]

        base_auc = roc_auc_score(y_test, base_pred)
        changed_auc = roc_auc_score(y_test, changed_pred)

        return {
            "feature_removed": feature_to_remove,
            "base_auc": base_auc,
            "changed_auc": changed_auc,
            "auc_change": changed_auc - base_auc,
            "status": "PASS" if abs(changed_auc - base_auc) < 0.05 else "FAIL"
        }

    def temporal_backtest(self, df, date_col, score_col, target_col):
        """
        Intended:
        Check model performance over time.

        FLAWS:
        - Uses an existing score column rather than model predictions.
        - Does not verify that score_col is available at prediction time.
        - Monthly grouping may hide weekly instability.
        """
        tmp = df.copy()
        tmp["month"] = pd.to_datetime(tmp[date_col]).dt.to_period("M").astype(str)

        rows = []
        for month, g in tmp.groupby("month"):
            if g[target_col].nunique() < 2:
                continue
            auc = roc_auc_score(g[target_col], g[score_col])
            rows.append({"month": month, "auc": auc, "n": len(g)})

        return pd.DataFrame(rows)
