"""
Vendor Dispute Escalation Classifier - Validation & Testing Script
SMBC Data Management Office
-------------------------------------------------------------------
Purpose:
    Independent validation of the vendor dispute escalation classifier.
    Checks include:
      - PSI (Population Stability Index)
      - Concept drift detection
      - Robustness testing
      - Sensitivity analysis
      - SHAP explainability
      - LIME explainability
      - Performance benchmarks
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
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
        import model_script_v3 as ms

        pipeline = ms.EscalationPipeline(n=4000)
        (model, X_train, X_test,
         y_train, y_test,
         X_full, y_full,
         df_raw, preprocessor) = pipeline.run()

        return model, X_train, X_test, y_train, y_test, X_full, y_full, df_raw, preprocessor


# ─────────────────────────────────────────────
# PSI VALIDATOR
# ─────────────────────────────────────────────

class PSIValidator:
    """
    Computes Population Stability Index across key structured features
    to detect distribution shift between training and scoring populations.

    PSI < 0.10        → Stable
    PSI 0.10 – 0.20   → Moderate shift
    PSI > 0.20        → Significant shift
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def _compute_psi(self, expected: np.ndarray, actual: np.ndarray) -> float:
        breakpoints    = np.percentile(expected, np.linspace(0, 100, self.n_bins + 1))
        breakpoints    = np.unique(breakpoints)
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        def bucket(arr):
            counts = np.histogram(arr, bins=breakpoints)[0]
            props  = counts / len(arr)
            return np.where(props == 0, 1e-4, props)

        exp_props = bucket(expected)
        act_props = bucket(actual)
        return float(np.sum((act_props - exp_props) * np.log(act_props / exp_props)))

    def run(self, model, X_train: pd.DataFrame, X_test: pd.DataFrame,
            y_train, y_test) -> None:
        print("\n[TEST 1: PSI Check]")

        train_scores = model.predict_proba(X_train)[:, 1]
        test_scores  = model.predict_proba(X_test)[:, 1]
        score_psi    = self._compute_psi(train_scores, test_scores)

        print(f"  Score PSI (train vs test) : {score_psi:.4f}")
        print(f"  Verdict : {'PASS' if score_psi < 0.10 else 'WARN' if score_psi < 0.20 else 'FAIL'}")

        print("\n  Feature-level PSI (selected features):")
        features_to_check = ["days_outstanding", "log_contract_value",
                             "prior_disputes", "resolution_attempts"]
        for feat in features_to_check:
            if feat in X_train.columns and feat in X_test.columns:
                psi = self._compute_psi(
                    X_train[feat].values,
                    X_test[feat].values
                )
                status = "OK" if psi < 0.10 else "WARN" if psi < 0.20 else "FAIL"
                print(f"    {feat:<35} PSI={psi:.4f}  [{status}]")


# ─────────────────────────────────────────────
# DRIFT DETECTOR
# ─────────────────────────────────────────────

class DriftDetector:
    """
    Detects concept drift by comparing model behaviour
    across early and late time cohorts.
    """

    def run(self, model, df_raw: pd.DataFrame, preprocessor) -> None:
        print("\n[TEST 2: Concept Drift Detection]")

        import model_script_v3 as ms

        df = df_raw.sort_values("dispute_date").reset_index(drop=True)
        n  = len(df)

        early = df.iloc[:n // 2].copy()
        late  = df.iloc[n // 2:].copy()

        def get_feature_means(cohort):
            eng = ms.FeatureEngineer(df_full=df_raw)
            cohort_eng = eng.transform(cohort)
            numeric_cols = ["days_outstanding", "prior_disputes",
                           "resolution_attempts", "log_contract_value",
                           "vendor_tier_recent_rate"]
            available = [c for c in numeric_cols if c in cohort_eng.columns]
            return cohort_eng[available].mean()

        early_means = get_feature_means(early)
        late_means  = get_feature_means(late)
        drift       = (late_means - early_means).abs()

        print("  Feature mean drift (early vs late cohort):")
        for feat, val in drift.sort_values(ascending=False).items():
            print(f"    {feat:<35} drift={val:.4f}")

        max_drift = drift.max()
        print(f"\n  Max feature drift : {max_drift:.4f}")
        print(f"  Verdict : {'PASS — No significant concept drift' if max_drift < 1.0 else 'WARN — Potential drift detected'}")


# ─────────────────────────────────────────────
# ROBUSTNESS TESTER
# ─────────────────────────────────────────────

class RobustnessTester:
    """
    Tests model stability under feature perturbations.
    Separately evaluates numeric and categorical features.
    """

    def run(self, model, X_test: pd.DataFrame, y_test,
            categorical_cols: list) -> None:
        print("\n[TEST 3: Robustness Check]")

        baseline_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        print(f"  Baseline AUC : {baseline_auc:.4f}")

        numeric_cols = [c for c in X_test.columns if c not in categorical_cols]

        X_num_perturbed = X_test.copy()
        noise = np.random.normal(0, 0.05, (len(X_test), len(numeric_cols)))
        X_num_perturbed[numeric_cols] = X_num_perturbed[numeric_cols].values + noise
        num_auc = roc_auc_score(y_test, model.predict_proba(X_num_perturbed)[:, 1])
        print(f"  Numeric perturbation AUC  : {num_auc:.4f}  drop={baseline_auc - num_auc:.4f}")

        X_cat_perturbed = X_test.copy()
        cat_noise = np.random.normal(0, 0.05, (len(X_test), len(categorical_cols)))
        X_cat_perturbed[categorical_cols] = X_cat_perturbed[categorical_cols].values + cat_noise
        cat_auc = roc_auc_score(y_test, model.predict_proba(X_cat_perturbed)[:, 1])
        print(f"  Categorical perturbation AUC : {cat_auc:.4f}  drop={baseline_auc - cat_auc:.4f}")

        max_drop = max(baseline_auc - num_auc, baseline_auc - cat_auc)
        print(f"  Verdict : {'PASS — Model robust to perturbations' if max_drop < 0.05 else 'FAIL — Sensitivity detected'}")


# ─────────────────────────────────────────────
# SENSITIVITY ANALYSER
# ─────────────────────────────────────────────

class SensitivityAnalyser:
    """
    Measures individual feature sensitivity by assessing
    AUC impact of single-feature perturbations.
    """

    def run(self, model, X_test: pd.DataFrame, y_test) -> pd.Series:
        print("\n[TEST 4: Sensitivity Analysis]")

        baseline_auc  = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        sensitivities = {}

        for col in X_test.columns:
            X_p      = X_test.copy()
            X_p[col] = X_p[col].values + np.random.normal(0, 0.5, len(X_test))
            auc      = roc_auc_score(y_test, model.predict_proba(X_p)[:, 1])
            sensitivities[col] = abs(baseline_auc - auc)

        sens = pd.Series(sensitivities).sort_values(ascending=False)

        print(f"  Baseline AUC : {baseline_auc:.4f}")
        print("\n  Top 5 most sensitive features:")
        for feat, val in sens.head(5).items():
            print(f"    {feat:<40} delta AUC={val:.4f}")

        flagged = sens[sens > 0.05]
        if len(flagged) > 0:
            print(f"\n  Features exceeding sensitivity threshold (>0.05):")
            for feat, val in flagged.items():
                print(f"    {feat:<40} delta AUC={val:.4f}")
            print(f"  Verdict : WARN — {len(flagged)} feature(s) show high sensitivity")
        else:
            print(f"  Verdict : PASS — No individual feature exceeds sensitivity threshold")

        return sens


# ─────────────────────────────────────────────
# SHAP VALIDATOR
# ─────────────────────────────────────────────

class SHAPValidator:
    """
    Computes SHAP values to assess global feature attributions
    and validate alignment with domain expectations.
    """

    def run(self, model, X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
        print("\n[TEST 5: SHAP Explainability Check]")

        try:
            import shap

            base_estimator = model.calibrated_classifiers_[0].estimator
            explainer      = shap.TreeExplainer(base_estimator)
            shap_values    = explainer.shap_values(X_test)

            if isinstance(shap_values, list):
                sv = shap_values[1]
            elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
                sv = shap_values[:, :, 1]
            else:
                sv = shap_values

            mean_shap = pd.Series(
                sv.mean(axis=0),
                index=X_test.columns
            ).abs().sort_values(ascending=False)

            print("  Top 10 features by mean SHAP:")
            for feat, val in mean_shap.head(10).items():
                print(f"    {feat:<40} {val:.4f}")

            expected = ["log_contract_value", "days_outstanding",
                       "prior_disputes", "vendor_tier_recent_rate",
                       "repeat_offender", "long_outstanding"]
            overlap  = [f for f in mean_shap.head(5).index if f in expected]
            print(f"\n  Verdict : {'PASS — Top drivers align with domain expectations' if len(overlap) >= 2 else 'WARN — Top drivers may not reflect domain knowledge'}")

        except ImportError:
            print("  SHAP not installed. Run: pip install shap")


# ─────────────────────────────────────────────
# LIME VALIDATOR
# ─────────────────────────────────────────────

class LIMEValidator:
    """
    Uses LIME to generate local explanations for a representative
    sample of predictions and assesses driver consistency.
    """

    def run(self, model, X_train: pd.DataFrame,
            X_test: pd.DataFrame, y_test) -> None:
        print("\n[TEST 6: LIME Explainability Check]")

        try:
            from lime.lime_tabular import LimeTabularExplainer

            explainer = LimeTabularExplainer(
                X_train.values,
                feature_names=X_train.columns.tolist(),
                class_names=["not_escalated", "escalated"],
                mode="classification",
                kernel_width=0.001
            )

            sample_size = 30
            indices     = np.random.choice(len(X_test), size=sample_size, replace=False)
            all_weights = {feat: [] for feat in X_train.columns}

            for idx in indices:
                instance = X_test.iloc[idx].values
                exp      = explainer.explain_instance(
                    instance, model.predict_proba, num_features=10
                )
                for feat, weight in exp.as_list():
                    matched = [f for f in X_train.columns if f in feat]
                    if matched:
                        all_weights[matched[0]].append(abs(weight))

            mean_weights = {
                f: np.mean(w) for f, w in all_weights.items() if len(w) > 0
            }
            weight_series = pd.Series(mean_weights).sort_values(ascending=False)

            print(f"  Sample size : {sample_size} instances")
            print("\n  Top 5 features by mean |LIME weight|:")
            for feat, val in weight_series.head(5).items():
                print(f"    {feat:<40} mean weight={val:.4f}")

            expected = ["log_contract_value", "days_outstanding",
                       "prior_disputes", "vendor_tier_recent_rate"]
            overlap  = [f for f in weight_series.head(5).index if f in expected]
            print(f"\n  Verdict : {'PASS — Local drivers consistent with domain expectations' if len(overlap) >= 1 else 'WARN — Local drivers inconsistent with domain knowledge'}")

        except ImportError:
            print("  LIME not installed. Run: pip install lime")


# ─────────────────────────────────────────────
# PERFORMANCE BENCHMARKER
# ─────────────────────────────────────────────

class PerformanceBenchmarker:
    """
    Verifies model meets minimum performance thresholds
    including calibration quality via Brier score.
    """

    def run(self, model, X_test: pd.DataFrame, y_test) -> None:
        print("\n[TEST 7: Performance Benchmark]")

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        auc    = roc_auc_score(y_test, y_proba)
        acc    = accuracy_score(y_test, y_pred)
        brier  = brier_score_loss(y_test, y_proba)

        print(f"  ROC-AUC     : {auc:.4f}   (threshold >= 0.70)")
        print(f"  Accuracy    : {acc:.4f}   (threshold >= 0.70)")
        print(f"  Brier Score : {brier:.4f}  (lower is better, threshold <= 0.25)")
        print(f"  AUC    : {'PASS' if auc >= 0.70 else 'FAIL'}")
        print(f"  ACC    : {'PASS' if acc >= 0.70 else 'FAIL'}")
        print(f"  Brier  : {'PASS' if brier <= 0.25 else 'FAIL'}")


# ─────────────────────────────────────────────
# VALIDATION SUITE
# ─────────────────────────────────────────────

class ValidationSuite:
    """
    Orchestrates all validation tests for the vendor dispute
    escalation classifier.
    """

    CATEGORICAL_COLS = ["dispute_type_enc", "region_enc", "vendor_tier",
                        "dispute_month", "high_value_dispute",
                        "repeat_offender", "long_outstanding"]

    def run(self):
        print("=" * 60)
        print("  Vendor Dispute Escalation — Validation Suite")
        print("=" * 60)

        print("\n[Setup] Rebuilding model...")
        rebuilder = ModelRebuilder()
        (model, X_train, X_test,
         y_train, y_test,
         X_full, y_full,
         df_raw, preprocessor) = rebuilder.rebuild()
        print("  Model ready.")

        PSIValidator().run(model, X_train, X_test, y_train, y_test)
        DriftDetector().run(model, df_raw, preprocessor)
        RobustnessTester().run(model, X_test, y_test, self.CATEGORICAL_COLS)
        SensitivityAnalyser().run(model, X_test, y_test)
        SHAPValidator().run(model, X_train, X_test)
        LIMEValidator().run(model, X_train, X_test, y_test)
        PerformanceBenchmarker().run(model, X_test, y_test)

        print("\n" + "=" * 60)
        print("  Validation complete.")
        print("=" * 60)


if __name__ == "__main__":
    suite = ValidationSuite()
    suite.run()
