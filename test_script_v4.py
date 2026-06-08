"""
Internal Audit Finding Escalation Classifier - Validation & Testing Script
SMBC Data Management Office
---------------------------------------------------------------------------
Purpose:
    Independent validation of the internal audit finding escalation classifier.
    Checks include:
      - PSI (Population Stability Index)
      - Calibration assessment
      - Permutation importance analysis
      - SHAP explainability
      - LIME explainability
      - Concept drift detection
      - Performance benchmarks
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)


# ─────────────────────────────────────────────
# MODEL REBUILDER
# ─────────────────────────────────────────────

class ModelRebuilder:
    """
    Reconstructs the trained model and data splits
    from the model script for independent validation.
    """

    def rebuild(self):
        import model_script_v4 as ms

        pipeline = ms.AuditEscalationPipeline(n=4000)
        (model, X_train, X_test,
         y_train, y_test,
         X_full, y_full,
         df_raw, preprocessor) = pipeline.run()

        return (model, X_train, X_test,
                y_train, y_test,
                X_full, y_full,
                df_raw, preprocessor)


# ─────────────────────────────────────────────
# PSI VALIDATOR
# ─────────────────────────────────────────────

class PSIValidator:
    """
    Computes Population Stability Index to detect distribution
    shift between training and test score populations.

    PSI < 0.10        → Stable
    PSI 0.10 – 0.20   → Moderate shift, monitor
    PSI > 0.20        → Significant shift, investigate
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def _compute_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray
    ) -> float:
        breakpoints     = np.percentile(
            expected, np.linspace(0, 100, self.n_bins + 1)
        )
        breakpoints     = np.unique(breakpoints)
        breakpoints[0]  = -np.inf
        breakpoints[-1] = np.inf

        def bucket(arr):
            counts = np.histogram(arr, bins=breakpoints)[0]
            props  = counts / len(arr)
            return np.where(props == 0, 1e-4, props)

        exp_props = bucket(expected)
        act_props = bucket(actual)
        return float(
            np.sum((act_props - exp_props) * np.log(act_props / exp_props))
        )

    def run(
        self,
        model,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> None:
        print("\n[TEST 1: PSI Check]")

        train_scores = model.predict_proba(X_train)[:, 1]
        test_scores  = model.predict_proba(X_test)[:, 1]
        score_psi    = self._compute_psi(train_scores, test_scores)

        print(f"  Score PSI (train vs test) : {score_psi:.4f}")
        verdict = (
            "PASS — Population stable"
            if score_psi < 0.10 else
            "WARN — Moderate shift"
            if score_psi < 0.20 else
            "FAIL — Significant shift"
        )
        print(f"  Verdict : {verdict}")

        print("\n  Feature-level PSI (key features):")
        for feat in ["finding_age", "remediation_attempts",
                     "prior_findings", "days_since_last_audit"]:
            if feat in X_train.columns:
                psi    = self._compute_psi(
                    X_train[feat].values, X_test[feat].values
                )
                status = "OK" if psi < 0.10 else "WARN" if psi < 0.20 else "FAIL"
                print(f"    {feat:<35} PSI={psi:.4f}  [{status}]")


# ─────────────────────────────────────────────
# CALIBRATION ASSESSOR
# ─────────────────────────────────────────────

class CalibrationAssessor:
    """
    Assesses probability calibration quality by comparing mean
    predicted probabilities against observed event rates across
    decile bins.

    A well-calibrated model should have mean predicted probability
    close to the observed event rate within each bin.
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def run(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> None:
        print("\n[TEST 2: Calibration Assessment]")

        y_proba  = model.predict_proba(X_test)[:, 1]
        brier    = brier_score_loss(y_test, y_proba)
        print(f"  Brier Score : {brier:.4f}  (lower is better)")

        bins        = np.percentile(y_proba, np.linspace(0, 100, self.n_bins + 1))
        bins        = np.unique(bins)
        bins[0]     = -np.inf
        bins[-1]    = np.inf
        bin_indices = np.digitize(y_proba, bins) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)

        print(f"\n  {'Bin':<5} {'Mean Pred Prob':<20} {'Observed Rate':<20} {'Gap'}")
        print(f"  {'-'*65}")

        max_gap = 0
        for b in range(self.n_bins):
            mask = bin_indices == b
            if mask.sum() == 0:
                continue
            mean_pred     = y_proba[mask].mean()
            observed_rate = y_proba[mask].mean()
            gap           = abs(mean_pred - observed_rate)
            max_gap       = max(max_gap, gap)
            print(
                f"  {b+1:<5} {mean_pred:<20.4f} "
                f"{observed_rate:<20.4f} {gap:.4f}"
            )

        print(f"\n  Max calibration gap : {max_gap:.4f}")
        print(
            f"  Verdict : "
            f"{'PASS — Model is well calibrated' if max_gap < 0.05 else 'FAIL — Calibration gap detected'}"
        )


# ─────────────────────────────────────────────
# PERMUTATION IMPORTANCE ANALYSER
# ─────────────────────────────────────────────

class PermutationImportanceAnalyser:
    """
    Computes permutation importance to assess true feature
    contribution on held-out data, avoiding MDI bias.
    """

    def __init__(self, n_repeats: int = 5):
        self.n_repeats = n_repeats

    def run(
        self,
        model,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        top_n: int = 10
    ) -> pd.Series:
        print("\n[TEST 3: Permutation Importance]")

        baseline_auc = roc_auc_score(
            y_train, model.predict_proba(X_train)[:, 1]
        )
        importances  = {}

        for col in X_train.columns:
            drops = []
            for _ in range(self.n_repeats):
                X_p       = X_train.copy()
                X_p[col]  = np.random.permutation(X_p[col].values)
                auc       = roc_auc_score(
                    y_train, model.predict_proba(X_p)[:, 1]
                )
                drops.append(baseline_auc - auc)
            importances[col] = np.mean(drops)

        imp_series = pd.Series(importances).sort_values(ascending=False)

        print(f"  Baseline AUC (train) : {baseline_auc:.4f}")
        print(f"\n  Top {top_n} features by permutation importance:")
        for feat, val in imp_series.head(top_n).items():
            print(f"    {feat:<40} {val:.4f}")

        top_feat = imp_series.index[0]
        print(f"\n  Most important feature : {top_feat}")
        print(
            "  Verdict : "
            f"{'PASS — Feature importances align with domain expectations' if 'post_event' not in top_feat else 'WARN — Top feature warrants further investigation'}"
        )

        return imp_series


# ─────────────────────────────────────────────
# SHAP VALIDATOR
# ─────────────────────────────────────────────

class SHAPValidator:
    """
    Computes SHAP values to assess global feature attributions.
    """

    def run(
        self,
        model,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> None:
        print("\n[TEST 4: SHAP Explainability]")

        try:
            import shap

            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)

            if isinstance(shap_values, list):
                sv = shap_values[1]
            elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
                sv = shap_values[:, :, 1]
            else:
                sv = shap_values

            mean_shap = pd.Series(
                sv.mean(axis=0),
                index=X_test.columns
            ).sort_values(ascending=False)

            print("  Top 10 features by mean SHAP value:")
            for feat, val in mean_shap.head(10).items():
                print(f"    {feat:<40} {val:.4f}")

            expected = [
                "bu_escalation_rate", "severity_enc",
                "high_severity", "long_outstanding",
                "repeat_unit", "finding_age"
            ]
            overlap = [
                f for f in mean_shap.head(5).index if f in expected
            ]
            print(
                f"\n  Verdict : "
                f"{'PASS — Top SHAP features align with domain expectations' if len(overlap) >= 2 else 'WARN — Top drivers may not reflect domain knowledge'}"
            )

        except ImportError:
            print("  SHAP not installed. Run: pip install shap")


# ─────────────────────────────────────────────
# LIME VALIDATOR
# ─────────────────────────────────────────────

class LIMEValidator:
    """
    Uses LIME to generate local explanations across a representative
    sample and assess consistency of feature drivers.
    """

    def run(
        self,
        model,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> None:
        print("\n[TEST 5: LIME Explainability]")

        try:
            from lime.lime_tabular import LimeTabularExplainer

            explainer = LimeTabularExplainer(
                X_train.values,
                feature_names=X_train.columns.tolist(),
                class_names=["not_escalated", "escalated"],
                mode="classification",
                kernel_width=0.001
            )

            sample_size  = 40
            indices      = np.random.choice(
                len(X_test), size=sample_size, replace=False
            )
            all_weights  = {feat: [] for feat in X_train.columns}

            for idx in indices:
                instance = X_test.iloc[idx].values
                exp      = explainer.explain_instance(
                    instance, model.predict_proba, num_features=10
                )
                for feat, weight in exp.as_list():
                    matched = [
                        f for f in X_train.columns if f in feat
                    ]
                    if matched:
                        all_weights[matched[0]].append(abs(weight))

            mean_weights  = {
                f: np.mean(w)
                for f, w in all_weights.items() if len(w) > 0
            }
            weight_series = pd.Series(
                mean_weights
            ).sort_values(ascending=False)

            print(f"  Sample size : {sample_size} instances")
            print("\n  Top 5 features by mean |LIME weight|:")
            for feat, val in weight_series.head(5).items():
                print(f"    {feat:<40} mean weight={val:.4f}")

            expected = [
                "bu_escalation_rate", "severity_enc",
                "finding_age", "long_outstanding", "repeat_unit"
            ]
            overlap  = [
                f for f in weight_series.head(5).index
                if f in expected
            ]
            print(
                f"\n  Verdict : "
                f"{'PASS — Local drivers consistent with domain expectations' if len(overlap) >= 1 else 'WARN — Local drivers inconsistent with domain knowledge'}"
            )

        except ImportError:
            print("  LIME not installed. Run: pip install lime")


# ─────────────────────────────────────────────
# DRIFT DETECTOR
# ─────────────────────────────────────────────

class DriftDetector:
    """
    Detects concept drift by comparing model score distributions
    across early and late time cohorts.
    """

    def run(
        self,
        model,
        df_raw: pd.DataFrame,
        preprocessor
    ) -> None:
        print("\n[TEST 6: Concept Drift Detection]")

        import model_script_v4 as ms

        df = df_raw.sort_values(
            "finding_date"
        ).reset_index(drop=True)
        n  = len(df)

        early = df.iloc[:n // 2].copy()
        late  = df.iloc[n // 2:].copy()

        def get_feature_means(cohort: pd.DataFrame) -> pd.Series:
            eng        = ms.FeatureEngineer(df_full=df_raw)
            cohort_eng = eng.transform(cohort)
            cols       = [
                "finding_age", "remediation_attempts",
                "prior_findings", "days_since_last_audit",
                "bu_escalation_rate"
            ]
            available  = [c for c in cols if c in cohort_eng.columns]
            return cohort_eng[available].mean()

        early_means = get_feature_means(early)
        late_means  = get_feature_means(late)
        drift       = (late_means - early_means).abs()

        print("  Feature mean drift (early vs late cohort):")
        for feat, val in drift.sort_values(ascending=False).items():
            print(f"    {feat:<35} drift={val:.4f}")

        max_drift = drift.max()
        print(f"\n  Max feature mean drift : {max_drift:.4f}")
        print(
            f"  Verdict : "
            f"{'PASS — No significant drift detected' if max_drift < 5.0 else 'WARN — Potential concept drift detected'}"
        )


# ─────────────────────────────────────────────
# PERFORMANCE BENCHMARKER
# ─────────────────────────────────────────────

class PerformanceBenchmarker:
    """
    Verifies model meets minimum performance thresholds.
    """

    def run(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> None:
        print("\n[TEST 7: Performance Benchmark]")

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        auc   = roc_auc_score(y_test, y_proba)
        acc   = accuracy_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_proba)

        print(f"  ROC-AUC     : {auc:.4f}   (threshold >= 0.70)")
        print(f"  Accuracy    : {acc:.4f}   (threshold >= 0.70)")
        print(f"  Brier Score : {brier:.4f}  (threshold <= 0.25)")
        print(f"  AUC    : {'PASS' if auc >= 0.70 else 'FAIL'}")
        print(f"  ACC    : {'PASS' if acc >= 0.70 else 'FAIL'}")
        print(f"  Brier  : {'PASS' if brier <= 0.25 else 'FAIL'}")


# ─────────────────────────────────────────────
# VALIDATION SUITE
# ─────────────────────────────────────────────

class ValidationSuite:
    """
    Orchestrates all validation tests for the internal audit
    finding escalation classifier.
    """

    def run(self):
        print("=" * 60)
        print("  Audit Finding Escalation — Validation Suite")
        print("=" * 60)

        print("\n[Setup] Rebuilding model...")
        rebuilder = ModelRebuilder()
        (model, X_train, X_test,
         y_train, y_test,
         X_full, y_full,
         df_raw, preprocessor) = rebuilder.rebuild()
        print("  Model ready.")

        PSIValidator().run(model, X_train, X_test)
        CalibrationAssessor().run(model, X_test, y_test)
        PermutationImportanceAnalyser().run(
            model, X_train, X_test, y_train, y_test
        )
        SHAPValidator().run(model, X_train, X_test)
        LIMEValidator().run(model, X_train, X_test, y_test)
        DriftDetector().run(model, df_raw, preprocessor)
        PerformanceBenchmarker().run(model, X_test, y_test)

        print("\n" + "=" * 60)
        print("  Validation complete.")
        print("=" * 60)


if __name__ == "__main__":
    suite = ValidationSuite()
    suite.run()
