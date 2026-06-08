
# Answer Key — Practice Pack Without Script Hints

## Model script issues

### 1. Post-event / unavailable-at-scoring features
Question these variables:
- `resolution_days`
- `investigation_outcome`
- `post_review_risk_score`
- `fast_resolution`
- `bad_outcome_flag`
- possibly `assigned_specialist_team`

Strong explanation:
The first thing to establish is the model scoring timestamp. If the model scores incoming cases before review or resolution, these variables would not be available and would leak future information.

### 2. Target encoding leakage
`segment_escalation_rate` and `channel_escalation_rate` are created using target information before the train/test split.

Correct approach:
Target encoding should be fitted only on training data. During cross-validation, it must be fitted only inside each training fold.

### 3. Feature engineering before split
`prepare_data()` fits the feature engineer on the full dataset before splitting.

This contaminates the holdout set because information from test rows has already influenced the engineered features.

### 4. Cross-validation issue
The CV uses shuffled `KFold`.

For event-time classification, this may be inappropriate because production usually means predicting future cases from past cases.

Better:
- time-based split
- walk-forward validation
- all preprocessing inside the CV pipeline

### 5. Text feature concern
`issue_text` may be valid if it is raw customer text available at intake.

It may be invalid if the text includes analyst notes, review notes, escalation wording added later, or outcome descriptions.

Good interview line:
I would not automatically reject text features, but I would validate the timestamp, source system, and whether text is frozen before scoring.

### 6. Unrealistically high performance
If the script produces very strong AUC, that is not automatically good.

It may indicate:
- target leakage
- post-event features
- random split optimism
- target encoding contamination

## Testing script issues

### 1. PSI implementation
Issues:
- bins are based on combined expected and actual data
- categorical/text PSI is not handled properly
- zero handling is too crude
- no sample size checks
- formula direction does not match the documented convention

Better:
- define bins using expected/reference data
- apply the same bins to actual/current data
- use epsilon for zero proportions
- handle categorical PSI separately
- output bucket-level table, not only one value

### 2. Prediction stability test
It only compares mean predicted score.

This can miss distribution shifts where the mean stays similar but the score distribution changes.

Better:
- score PSI
- KS statistic
- score decile distribution
- high-score tail concentration
- weekly/monthly performance trend if labels are available

### 3. Robustness test
Issues:
- noise level is too small and absolute, not feature-scaled
- binary variables may become decimals
- categorical/text features are not tested
- only one random seed
- no repeated simulation

Better:
- perturb continuous features based on realistic measurement error
- keep binary/categorical features valid
- repeat across seeds
- compare AUC, PR-AUC, recall, and score rank stability

### 4. Feature removal sensitivity
Replacing a feature with zero is not the same as removing a feature.

Problems:
- zero may be meaningless or out-of-distribution
- invalid for categorical/text fields
- does not retrain model
- measures masking impact, not full dependency

Better:
- permutation importance
- retrain without the feature
- SHAP/global feature importance
- compare performance degradation

### 5. Temporal backtest
It uses an arbitrary `score_col`.

A validator should check whether the score is the model prediction or a leaked operational score.

Better:
- use actual model predicted probabilities
- test by scoring period
- consider label lag
- use weekly/monthly trend depending on business cadence

## Strong interview structure

When reviewing the model script:

1. Identify model purpose and prediction timestamp.
2. Check whether each feature exists at that timestamp.
3. Look for direct and indirect target leakage.
4. Check whether feature engineering is fitted before the split.
5. Check split strategy.
6. Check whether CV is fold-safe and time-aware.
7. Run the script and investigate suspiciously high performance.
8. Inspect top features or coefficients.
9. Challenge text features based on source and timing.
10. Recommend corrected validation approach.

When reviewing the testing script:

1. Check PSI formula and binning logic.
2. Check numeric vs categorical PSI treatment.
3. Check whether stability means full distribution stability.
4. Check whether robustness perturbations are realistic.
5. Check whether binary/categorical variables remain valid.
6. Check whether sensitivity test is masking or retraining.
7. Check whether temporal backtest uses real model predictions.
