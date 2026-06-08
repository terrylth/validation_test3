
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class EscalationDataBuilder:
    def __init__(self, n_rows=12000, random_state=42):
        self.n_rows = n_rows
        self.random_state = random_state

    def build(self):
        rng = np.random.default_rng(self.random_state)

        start_date = pd.Timestamp("2024-01-01")
        event_date = start_date + pd.to_timedelta(
            rng.integers(0, 420, size=self.n_rows), unit="D"
        )

        customer_segment = rng.choice(
            ["retail", "sme", "corporate", "private"],
            size=self.n_rows,
            p=[0.50, 0.25, 0.20, 0.05]
        )
        channel = rng.choice(["email", "call", "branch", "app"], size=self.n_rows)
        region = rng.choice(["SG", "MY", "HK", "ID"], size=self.n_rows)
        priority_flag = rng.binomial(1, 0.18, size=self.n_rows)

        account_age_days = rng.integers(30, 3000, size=self.n_rows)
        num_contacts_30d = rng.poisson(2.2, size=self.n_rows)
        avg_response_mins_90d = rng.gamma(2.0, 80.0, size=self.n_rows)

        assigned_specialist_team = rng.choice(
            ["general", "complaints", "technical", "legal"],
            size=self.n_rows,
            p=[0.65, 0.18, 0.12, 0.05]
        )

        text_templates = np.array([
            "customer asked for status update",
            "payment delay and repeated follow up",
            "complaint about unresolved case",
            "customer mentions regulator and formal complaint",
            "account access problem",
            "pricing dispute and escalation request",
            "normal inquiry on transaction",
            "urgent issue unresolved for days"
        ])
        issue_text = rng.choice(text_templates, size=self.n_rows)

        logit = (
            -3.2
            + 0.9 * priority_flag
            + 0.35 * (num_contacts_30d > 4)
            + 0.55 * np.isin(channel, ["email", "call"]).astype(int)
            + 0.45 * np.isin(customer_segment, ["corporate", "private"]).astype(int)
            + 0.75 * pd.Series(issue_text).str.contains("complaint|regulator|urgent").astype(int).values
            + 0.25 * (avg_response_mins_90d > 220)
        )

        prob = 1 / (1 + np.exp(-logit))
        escalated = rng.binomial(1, prob)

        resolution_days = np.where(
            escalated == 1,
            rng.normal(9, 3, size=self.n_rows),
            rng.normal(2, 1, size=self.n_rows)
        )
        resolution_days = np.clip(resolution_days, 0.5, None)

        investigation_outcome = np.where(
            escalated == 1,
            rng.choice(["confirmed_escalation", "management_review"], size=self.n_rows, p=[0.75, 0.25]),
            rng.choice(["closed_no_issue", "duplicate", "info_provided"], size=self.n_rows, p=[0.55, 0.25, 0.20])
        )

        post_review_risk_score = (
            20
            + 50 * escalated
            + 6 * priority_flag
            + rng.normal(0, 8, size=self.n_rows)
        )

        df = pd.DataFrame({
            "case_id": np.arange(1, self.n_rows + 1),
            "event_date": event_date,
            "customer_segment": customer_segment,
            "channel": channel,
            "region": region,
            "priority_flag": priority_flag,
            "account_age_days": account_age_days,
            "num_contacts_30d": num_contacts_30d,
            "avg_response_mins_90d": avg_response_mins_90d,
            "assigned_specialist_team": assigned_specialist_team,
            "issue_text": issue_text,
            "resolution_days": resolution_days,
            "investigation_outcome": investigation_outcome,
            "post_review_risk_score": post_review_risk_score,
            "escalated": escalated
        })

        df.loc[(df["region"] == "ID") & (rng.random(self.n_rows) < 0.25), "avg_response_mins_90d"] = np.nan

        return df


class EscalationFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.segment_rates_ = None
        self.channel_rates_ = None

    def fit(self, X, y=None):
        tmp = X.copy()

        if "escalated" in tmp.columns:
            self.segment_rates_ = tmp.groupby("customer_segment")["escalated"].mean().to_dict()
            self.channel_rates_ = tmp.groupby("channel")["escalated"].mean().to_dict()
        elif y is not None:
            tmp["escalated"] = y
            self.segment_rates_ = tmp.groupby("customer_segment")["escalated"].mean().to_dict()
            self.channel_rates_ = tmp.groupby("channel")["escalated"].mean().to_dict()

        return self

    def transform(self, X):
        df = X.copy()

        df["event_month"] = pd.to_datetime(df["event_date"]).dt.month
        df["event_dayofweek"] = pd.to_datetime(df["event_date"]).dt.dayofweek

        df["fast_resolution"] = (df["resolution_days"] <= 3).astype(int)
        df["bad_outcome_flag"] = df["investigation_outcome"].isin(
            ["confirmed_escalation", "management_review"]
        ).astype(int)

        df["segment_escalation_rate"] = df["customer_segment"].map(self.segment_rates_).fillna(0)
        df["channel_escalation_rate"] = df["channel"].map(self.channel_rates_).fillna(0)

        df["contains_escalation_word"] = df["issue_text"].str.contains(
            "escalation|regulator|formal complaint|urgent", case=False, na=False
        ).astype(int)

        return df


class EscalationModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.feature_engineer = EscalationFeatureEngineer()
        self.pipeline = None

    def prepare_data(self, df):
        df_fe = self.feature_engineer.fit_transform(df)

        features = [
            "customer_segment", "channel", "region", "assigned_specialist_team",
            "priority_flag", "account_age_days", "num_contacts_30d", "avg_response_mins_90d",
            "event_month", "event_dayofweek",
            "fast_resolution", "bad_outcome_flag",
            "post_review_risk_score",
            "segment_escalation_rate", "channel_escalation_rate",
            "contains_escalation_word",
            "issue_text"
        ]

        X = df_fe[features]
        y = df_fe["escalated"]

        return X, y

    def split_data(self, X, y):
        return train_test_split(
            X, y, test_size=0.25, random_state=self.random_state, stratify=y
        )

    def build_pipeline(self):
        numeric_features = [
            "priority_flag", "account_age_days", "num_contacts_30d", "avg_response_mins_90d",
            "event_month", "event_dayofweek",
            "fast_resolution", "bad_outcome_flag", "post_review_risk_score",
            "segment_escalation_rate", "channel_escalation_rate", "contains_escalation_word"
        ]

        categorical_features = [
            "customer_segment", "channel", "region", "assigned_specialist_team"
        ]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]), numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
                ("text", TfidfVectorizer(max_features=50, ngram_range=(1, 2)), "issue_text")
            ],
            remainder="drop"
        )

        self.pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000, C=10.0))
        ])

        return self.pipeline

    def cross_validate(self, X, y):
        if self.pipeline is None:
            self.build_pipeline()

        cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(self.pipeline, X, y, cv=cv, scoring="roc_auc")

        return scores

    def train(self, X_train, y_train):
        if self.pipeline is None:
            self.build_pipeline()

        self.pipeline.fit(X_train, y_train)
        return self.pipeline

    def evaluate(self, X_test, y_test):
        y_proba = self.pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        return {
            "roc_auc": roc_auc_score(y_test, y_proba),
            "pr_auc": average_precision_score(y_test, y_proba),
            "classification_report": classification_report(y_test, y_pred)
        }


if __name__ == "__main__":
    data_builder = EscalationDataBuilder()
    df = data_builder.build()

    trainer = EscalationModelTrainer()

    X, y = trainer.prepare_data(df)

    cv_scores = trainer.cross_validate(X, y)
    print("CV ROC-AUC:", cv_scores)
    print("Mean CV ROC-AUC:", cv_scores.mean())

    X_train, X_test, y_train, y_test = trainer.split_data(X, y)

    trainer.train(X_train, y_train)

    results = trainer.evaluate(X_test, y_test)

    print("Holdout ROC-AUC:", results["roc_auc"])
    print("Holdout PR-AUC:", results["pr_auc"])
    print(results["classification_report"])
